import random

import torch
import torch.nn.functional as F


class StereoRobustAug:
    """
    Robust augmentation for real-world stereo reconstruction.
    Simulates residual rectification errors: vertical shift (±1-2 px) and
    small rotation (±0.1-0.3 degrees).
    """

    def __init__(
        self,
        max_shift=2,  # Maximum vertical shift in pixels.
        max_angle=0.2,  # Maximum rotation angle in degrees.
        prob=0.5,  # Probability of applying augmentation.
        apply_shift=True,
        apply_rotate=True,
    ):
        self.max_shift = max_shift
        self.max_angle = max_angle
        self.prob = prob
        self.apply_shift = apply_shift
        self.apply_rotate = apply_rotate
        self.current_shift = 0.0
        self.current_angle = 0.0

    def __call__(self, img_l, img_r, gt_r=None):
        if random.random() > self.prob:
            return (img_l, img_r, gt_r) if gt_r is not None else (img_l, img_r)

        # Sample perturbation parameters.
        self.current_shift = (
            random.uniform(-self.max_shift, self.max_shift) if self.apply_shift else 0.0
        )
        self.current_angle = (
            random.uniform(-self.max_angle, self.max_angle)
            if self.apply_rotate
            else 0.0
        )

        # Apply the same transform to the right input and right GT.
        img_r_aug = self._apply_transform(img_r, self.current_shift, self.current_angle)
        if gt_r is not None:
            gt_r_aug = self._apply_transform(
                gt_r, self.current_shift, self.current_angle
            )
            return img_l, img_r_aug, gt_r_aug
        else:
            return img_l, img_r_aug

    def _apply_transform(self, img, shift, angle):
        if abs(shift) < 1e-6 and abs(angle) < 1e-6:
            return img

        is_seq = img.dim() == 5
        is_single = img.dim() == 3

        if is_single:
            img = img.unsqueeze(0)
        elif is_seq:
            B, T, C, H, W = img.shape
            img = img.view(B * T, C, H, W)

        N, C, H, W = img.shape
        device = img.device

        # 1. Build a rotation matrix around the image center.
        angle_rad = torch.tensor(angle * 3.1415926535 / 180.0, device=device)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Translate to center -> rotate -> translate back.
        # Final affine matrix: [cos, -sin, tx; sin, cos, ty].
        tx = (1 - cos_a) * W / 2 + sin_a * H / 2
        ty = -sin_a * W / 2 + (1 - cos_a) * H / 2

        theta = torch.zeros(N, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = tx / (W / 2)  # Normalize to [-1,1].
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = ty / (H / 2)

        # Rotate first.
        grid_rot = F.affine_grid(theta, img.shape, align_corners=True)
        img_rot = F.grid_sample(
            img, grid_rot, mode="bilinear", padding_mode="border", align_corners=True
        )

        # 2. Then apply vertical shift.
        if abs(shift) > 1e-6:
            shift_norm = 2.0 * shift / (H - 1) if H > 1 else 0.0
            theta_shift = torch.zeros(N, 2, 3, device=device)
            theta_shift[:, 0, 0] = 1.0
            theta_shift[:, 1, 1] = 1.0
            theta_shift[:, 1, 2] = shift_norm

            grid_shift = F.affine_grid(theta_shift, img.shape, align_corners=True)
            img_out = F.grid_sample(
                img_rot,
                grid_shift,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        else:
            img_out = img_rot

        # Restore the original shape.
        if is_single:
            img_out = img_out.squeeze(0)
        elif is_seq:
            img_out = img_out.view(B, T, C, H, W)

        return img_out


class StereoAugmentation:
    """Stereo-pair augmentation with mild rectification errors and synchronized GT."""

    def __init__(
        self,
        vertical_shift_range=(-3, 3),  # Vertical shift range in pixels.
        rotation_range=(-1.0, 1.0),  # Rotation angle range in degrees.
        vertical_scale_range=(0.98, 1.02),  # Vertical scale range.
        prob=0.5,
    ):  # Probability of applying augmentation.
        self.vertical_shift_range = vertical_shift_range
        self.rotation_range = rotation_range
        self.vertical_scale_range = vertical_scale_range
        self.prob = prob

        # Store current augmentation parameters for synchronized GT augmentation.
        self.current_params = None

    def __call__(self, img_l, img_r, gt_r=None):
        """
        Args:
            img_l: left image, shaped (C,H,W), (B,C,H,W), or sequence (B,T,C,H,W)
            img_r: right image, shaped (C,H,W), (B,C,H,W), or sequence (B,T,C,H,W)
            gt_r: optional right GT with the same supported layouts
        Returns:
            Augmented left image, right image, and right GT when provided.
        """
        if random.random() > self.prob:
            if gt_r is not None:
                return img_l, img_r, gt_r
            return img_l, img_r

        # Sample one augmentation parameter set.
        self._sample_augmentation_params()

        # Apply augmentation to the right image.
        img_r_aug = self._apply_misalignment(img_r, use_current_params=True)

        # If GT is provided, augment it with the same parameters.
        gt_r_aug = None
        if gt_r is not None:
            gt_r_aug = self._apply_misalignment(gt_r, use_current_params=True)

        if gt_r is not None:
            return img_l, img_r_aug, gt_r_aug
        return img_l, img_r_aug

    def _sample_augmentation_params(self):
        """Sample one augmentation parameter set."""
        aug_type = random.choice(["shift", "rotate", "scale", "combined"])

        self.current_params = {
            "type": aug_type,
            "shift": random.uniform(*self.vertical_shift_range),
            "angle": random.uniform(*self.rotation_range),
            "scale": random.uniform(*self.vertical_scale_range),
        }

    def _apply_misalignment(self, img, use_current_params=False):
        """
        Apply rectification error to an image while supporting multiple layouts.
        Args:
            img: (C,H,W), (B,C,H,W), or (B,T,C,H,W)
            use_current_params: whether to reuse previously sampled parameters
        """
        is_sequence = img.dim() == 5  # (B, T, C, H, W)
        is_single = img.dim() == 3  # (C, H, W)

        # Convert to (B, C, H, W).
        if is_single:
            img = img.unsqueeze(0)  # (1, C, H, W)
        elif is_sequence:
            B, T, C, H, W = img.shape
            img = img.view(B * T, C, H, W)  # (B*T, C, H, W)

        # Resample if current parameters should not be reused.
        if not use_current_params or self.current_params is None:
            self._sample_augmentation_params()

        params = self.current_params
        aug_type = params["type"]

        # Apply augmentation.
        if aug_type == "shift":
            img = self._vertical_shift(img, params["shift"])
        elif aug_type == "rotate":
            img = self._small_rotation(img, params["angle"])
        elif aug_type == "scale":
            img = self._vertical_scale(img, params["scale"])
        else:  # combined
            img = self._vertical_shift(img, params["shift"])
            if random.random() > 0.5:
                img = self._small_rotation(img, params["angle"])

        # Restore the original shape.
        if is_single:
            img = img.squeeze(0)
        elif is_sequence:
            img = img.view(B, T, C, H, W)

        return img

    def _vertical_shift(self, img, shift):
        """Apply vertical shift."""
        shift_px = int(shift)

        if shift_px == 0:
            return img

        B, C, H, W = img.shape

        # Build shift matrix.
        theta = (
            torch.tensor(
                [
                    [1, 0, 0],
                    [0, 1, 2.0 * shift_px / H],  # Normalize to [-1, 1].
                ],
                dtype=torch.float32,
                device=img.device,
            )
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )

        grid = F.affine_grid(theta, img.size(), align_corners=False)
        img_shifted = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        return img_shifted

    def _small_rotation(self, img, angle):
        """Apply small-angle rotation."""
        angle_rad = angle * 3.14159 / 180.0

        B, C, H, W = img.shape

        cos_a = torch.cos(torch.tensor(angle_rad))
        sin_a = torch.sin(torch.tensor(angle_rad))

        # Rotation matrix.
        theta = (
            torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                dtype=torch.float32,
                device=img.device,
            )
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )

        grid = F.affine_grid(theta, img.size(), align_corners=False)
        img_rotated = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        return img_rotated

    def _vertical_scale(self, img, scale):
        """Apply vertical scaling."""
        B, C, H, W = img.shape

        # Scale only along the vertical direction.
        theta = (
            torch.tensor(
                [[1, 0, 0], [0, scale, 0]], dtype=torch.float32, device=img.device
            )
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )

        grid = F.affine_grid(theta, img.size(), align_corners=False)
        img_scaled = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        return img_scaled

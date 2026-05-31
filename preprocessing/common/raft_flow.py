"""RAFT optical-flow helpers shared by interpolation pipelines."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from runtime import bootstrap_interpolation_runtime

bootstrap_interpolation_runtime()

from raft import RAFT
from utils.utils import InputPadder


def load_raft_model(model_path: str, device: str):
    """Load the RAFT model used by preprocessing interpolation scripts."""
    args = argparse.Namespace()
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    args.dropout = 0.0
    args.corr_levels = 4
    args.corr_radius = 4
    args.hidden_dims = [128, 128, 96, 64, 32]

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.module
    model.to(device)
    model.eval()
    return model


def compute_optical_flow(model, img1: np.ndarray, img2: np.ndarray, device: str) -> np.ndarray:
    """Compute HWC optical flow from img1 to img2."""
    tensor1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None].to(device)
    tensor2 = torch.from_numpy(img2).permute(2, 0, 1).float()[None].to(device)

    padder = InputPadder(tensor1.shape)
    tensor1_pad, tensor2_pad = padder.pad(tensor1, tensor2)

    with torch.no_grad():
        _, flow = model(tensor1_pad, tensor2_pad, iters=20, test_mode=True)

    flow = padder.unpad(flow)[0].cpu().numpy()
    return flow.transpose(1, 2, 0)


def compute_bidirectional_flow(model, img1: np.ndarray, img2: np.ndarray, device: str):
    """Return forward and backward RAFT flow for a pair of frames."""
    flow_forward = compute_optical_flow(model, img1, img2, device)
    flow_backward = compute_optical_flow(model, img2, img1, device)
    return flow_forward, flow_backward

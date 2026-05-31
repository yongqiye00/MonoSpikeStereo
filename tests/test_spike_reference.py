import numpy as np

from dataset.spike_reference import compute_mt_reference


def test_mt_reference_uses_one_modulation_period() -> None:
    spikes = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8).reshape(8, 1, 1)

    ref = compute_mt_reference(
        spikes,
        window_size=2,
        frame_count=2,
        reference_window_size=4,
    )

    assert ref.shape == (2, 1, 1)
    np.testing.assert_allclose(ref[:, 0, 0], np.array([0.75, 0.75], dtype=np.float32))


def test_mt_reference_uses_trailing_period_after_warmup() -> None:
    spikes = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8).reshape(10, 1, 1)

    ref = compute_mt_reference(
        spikes,
        window_size=2,
        frame_count=3,
        reference_window_size=4,
    )

    assert ref.shape == (3, 1, 1)
    np.testing.assert_allclose(ref[:, 0, 0], np.array([0.75, 0.75, 0.5], dtype=np.float32))

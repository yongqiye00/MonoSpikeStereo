# Preprocessing Configs

Run:

```bash
python preprocessing/run_preprocessing.py --pipeline tartanair_interpolate \
    --config preprocessing/config/tartanair_interpolate.yaml

python preprocessing/run_preprocessing.py --pipeline tartanair_simulation \
    --config preprocessing/config/tartanair_simulation.yaml
```

## `tartanair_interpolate.yaml`

| Key | Meaning |
| --- | --- |
| `input_root` | Root directory of original TartanAir image sequences. |
| `output_root` | Directory for interpolated frame groups. |
| `raft_model_path` | RAFT checkpoint path. The default is `preprocessing/RAFT/models/raft-things.pth`. |
| `num_interpolations` | Number of intermediate frames inserted between two source frames. |
| `crop_size` | Center crop size in `HxW` format. |
| `max_frames` | Optional cap on source frames per sequence; `null` means no cap. |
| `group_size` | Number of source frames per interpolation group. |
| `max_groups` | Maximum groups processed per sequence. |
| `group_skip` | Stride between neighboring source-frame groups. |
| `use_bidirectional_flow` | Use forward and backward RAFT flow for interpolation. |

## `tartanair_simulation.yaml`

| Key | Meaning |
| --- | --- |
| `pipeline` | Pipeline name for the unified preprocessing runner. |
| `dataset` | Dataset label for readability. |
| `params.root` | Interpolated TartanAir root directory. |
| `params.out` | Output directory for generated NPZ files. |
| `params.threshold` | Exposed threshold value; the LCD path samples per-sequence values internally. |
| `params.length` | Shared CLI value; the LCD path controls temporal expansion internally. |
| `params.noise` | Shared CLI noise switch. |
| `params.noise_level` | Shared CLI noise level. |
| `params.crop` | Center crop size in `WxH` format. |
| `params.random_crop` | Random crop switch; the public LCD path uses center crops. |
| `params.lcd` | Enables the LCD mixed-spike simulation path. |
| `params.save_spike_sequences` | Save raw `spk` arrays in generated NPZ files. |

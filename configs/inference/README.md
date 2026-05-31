# Inference Configs

Run:

```bash
python inference.py --config configs/inference/inference.yaml
```

| Key | Meaning |
| --- | --- |
| `npz_path` | Directory or file containing raw-spike NPZ data. |
| `checkpoint` | Model checkpoint path. |
| `output_dir` | Directory for saved inference results. |
| `save_method_name` | Name prefix used by the output saver. |
| `name` | Experiment/model name. |
| `net_type` | Model family selector. The public preset uses `flow`. |
| `device` | `cuda` or `cpu`. |
| `checkpoint_dir` | Fallback checkpoint search directory. |
| `batch_size` | Inference batch size. |
| `num_workers` | DataLoader worker count. |
| `sequence_length` | Number of reconstructed frames per sample; non-positive values use the full sequence. |
| `state_history` | Number of recurrent/history states kept by the model wrapper. |
| `crop_size` | Center crop size in `HxW` format. |
| `max_files` | Optional cap on input NPZ files; `0` means no cap. |
| `save_every_n_frames` | Save one frame every N frames. |
| `ref_source` | `spk` computes `mt` from spikes; `stored` reads stored `mt`; `zeros` uses a zero reference. |
| `spike_reg` | Regularization used while splitting raw spikes into left/right streams. |
| `save_inputs` | Save reconstructed left/right inputs. |
| `save_gt` | Save ground truth if present in the NPZ. |
| `export_videos` | Export saved frames as videos. |
| `split_progress` | Show per-sample spike splitting progress. |

# Test Configs

Run:

```bash
python test.py --config configs/test/test.yaml
```

| Key | Meaning |
| --- | --- |
| `npz_path` | Directory or file containing test NPZ data. |
| `checkpoint` | Model checkpoint path. |
| `output_dir` | Directory for saved predictions and optional inputs/GT. |
| `save_method_name` | Name prefix used by the output saver. |
| `name` | Experiment/model name. |
| `net_type` | Model family selector. The public preset uses `flow`. |
| `device` | `cuda` or `cpu`. |
| `checkpoint_dir` | Fallback checkpoint search directory. |
| `batch_size` | Test batch size. |
| `num_workers` | DataLoader worker count. |
| `sequence_length` | Number of frames evaluated per sample. |
| `state_history` | Number of recurrent/history states kept by the model wrapper. |
| `crop_size` | Center crop size in `HxW` format. |
| `max_test_batches` | Optional cap on test batches; `0` means no cap. |
| `save_every_n_frames` | Save one frame every N frames. |
| `pred_only` | Save predictions only unless input/GT saving is enabled. |
| `save_inputs_gt` | Save input and ground-truth frames along with predictions. |

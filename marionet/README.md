To train a MarioNette model, run `python scripts/train.py --checkpoint_dir
out_dir --data data_dir`, where `data_dir` contains your dataset, with each
frame named `#.png`. Optionally, pass a `--layer_size` flag to specify the
anchor grid resolution, `--num_layers` to specify the number of layers, and
`--num_classes` to specify the number of dictionary elements.

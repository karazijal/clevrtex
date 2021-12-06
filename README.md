# ClevrTex: A Texture-Rich Benchmark for Unsupervised Multi-Object Segmentation
This repository contains experiment code for [ClevrTex: A Texture-Rich Benchmark for Unsupervised Multi-Object Segmentation](https://www.robots.ox.ac.uk/~vgg/research/clevrtex/)
paper. For dataset generation, see [here](https://github.com/karazijal/clevrtex-generation).

## Getting started

Set up environment
```bash
conda env create -f environment.yml
```

Download datasets and place them into `ool_data/<dataset>` folders. For ClevrTex, simply download and unpack
data of interest. For CLEVR, see intructions [here](https://github.com/deepmind/multi_object_datasets). 
We simply preprocessed CLEVR data by building numpy iterator and exporing it as `{index}.npz` files. 
See bellow for code snippet to do this.

## Running experiments
To run training for different models use e.g.:
```bash
python experiments/slota.py run --data 'clevr-crop-(128,128)' --tag slot-attention
```
```bash
python experiments/gnm.py run --data 'clevrtex:full-crop-(128,128)' --tag gnm-on-clevrtex --std 0.5
```
```bash
python experiments/genesisv2.py run --data 'clevrtex:full-crop-(128,128)' --tag genesisv2 --g_goal 0.5 --seed random
```
```bash
python experiments/space.py run --data 'clevr-crop-(128,128)' --tag space-on-clevr
```
```bash
python experiments/spair.py run --large --bg_dim 1 --std 0.15 --data 'clevr-crop-(128,128)' --tag spair-on-clevr
python experiments/spair.py run --large --bg_dim 4 --std 'itr_lin(50000,0.5,0.15)' --data 'clevrtex:full-crop-(128,128)' --tag spair-on-clevrtex --prior 'itr_lin(50000,1e-4,.99)'
```
```bash
python experiments/dtisprites.py run --data 'clevr-crop-(128,128)' --slots 10 --tag baseline
```
```bash
python experiments/iodine.py run --batch_size 8 --gc 200 --ddp --data 'clevr-crop-(128,128)' --drop-last-batch --K 11 --seed random --gpus '[0,1,2,3]'
```
```bash
python experiments/monet.py run --gc 200 --data 'clevr-crop-(128,128)' --tag baseline-repro --batch_size 21 --ddp --gpus '[0,1,2]'
```
```bash
python experiments/effmorl.py run --batch_size 8 --data 'clevr-crop-(128,128)' --drop-last-batch --ddp --gpus '[0,1,2,3]'
```

### Repository
This reporsitory implements ways to conduct experiments on various models.
Common experimentation framework relies [pytorch_lightning](https://pytorch-lightning.readthedocs.io)
with [python-fire](https://github.com/google/python-fire) to turn class kwargs into command line arguments.
As such, considering checking out [parent class for all experiments](experiments/framework/base_experiment.py) especially
`run()` method to see the plumbing and various running options. In briefs, certain options might be of interest:
 - `--gpus '[1]'` will select gpu 1 to run on.
 - `--ddp` will force the model to run DDP 
 - `--workers 4` will use 4 workers in  the dataloader
 - `--gc 200` will _force_ the model to run GC every 200 iterations. This should not be needed, but there might _be_ some
    wierdness with built-in data collections retaining tensor references for longer than needed. If hitting against the upper
   limit of GPU memory, consider setting this to non-zero value.
   
For changing the datasets (assuming it has been downloaded), pass `--data 'clevr-crop-(128,128)'` for clevr
or `--data 'clevrtex:<variant>-crop-(128,128)'` for CLEVRTEX and variants.

### Evaluation
We evaluation using ClevrTex benchmark evaluator rather than test method. Please example evaluation.ipynb notebook
how to load and run models.


### Third-party code

Repository contains the following third-party code from these repositories. The code has been modified and adjusted
to lower number of dependencies, use common framework and provide metrics and additional outputs.
Please check Licenses in respective [folders](ool/picture/models/thirdparty/).
 - DTI-Sprites (MIT License): [paper](https://arxiv.org/abs/2104.14575) [code](https://github.com/monniert/dti-sprites/)
 - Efficient MORL (MIT License): [paper](http://proceedings.mlr.press/v139/emami21a.html) [code](https://github.com/pemami4911/EfficientMORL)
 - Genesis V2 (GPLv3 License): [paper](https://arxiv.org/abs/2104.09958v2) [code](https://github.com/applied-ai-lab/genesis)
 - GNM (GPLv3 License): [paper](https://arxiv.org/abs/2010.12152) [code](https://github.com/JindongJiang/GNM)
 - SPACE (GPLv3 License): [paper](https://arxiv.org/abs/2001.02407) [code](https://github.com/zhixuan-lin/SPACE)
 - Slot Attention (MIT): [paper](https://arxiv.org/abs/2006.15055) [code](https://github.com/lucidrains/slot-attention) [orginal code](https://github.com/google-research/google-research/tree/master/slot_attention)
 - MarioNette* (MIT): [paper](https://arxiv.org/abs/2104.14553) [code](https://github.com/dmsm/MarioNette)

In addition, we use re-implemetation of following models, with other implementation and/or original code used for verification.
 - IODINE [paper](http://proceedings.mlr.press/v97/greff19a.html) [original code](https://github.com/deepmind/deepmind-research/tree/master/iodine)
 - MONet [paper](https://arxiv.org/abs/1901.11390) 
 - SPAIR [paper](http://e2crawfo.github.io/pdfs/spair_aaai_2019.pdf) [original code](https://github.com/e2crawfo/auto_yolo)

*- We used a version of the source code kindly shared with before the official public release.

#### Converting CLEVR snippet
Here is how the CLEVR was converted.
```python
output = Path('to/your/data/output')
data = clevr_tf_dataset # Multi-object-datasets repository

m = {}
for i, d in enumerate(tqdm(data.as_numpy_iterator(), desc=f"Converting CLEVR")):
    batch_id = i // 10000
    opath = output / str(batch_id)
    opath.mkdir(exist_ok=True)
    opath = opath / str(i)
    m[i] = f"{batch_id}/{i}.npz"
    np.savez_compressed(str(opath), **d)
with (output / 'manifest_ind.json').open('w') as outf:
        outf.write(json.dumps(m))
```

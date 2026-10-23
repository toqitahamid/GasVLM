# GasVLM

GasVLM is a grounded vision-language model for optical gas imaging (OGI). OGI uses
thermal infrared cameras to see gas plumes that are invisible to the eye. GasVLM takes
one thermal OGI frame and a question, and it writes a text answer. The answer can
contain `[SEG]` tokens. Each `[SEG]` token drives a mask decoder that outputs a binary
mask of the gas plume or the gas source. For composition questions, the answer text
gives the CH4 and CO2 percentages and the CH4/CO2 ratio class.

GasVLM handles three task levels:

- **RES** (referring expression segmentation): the answer has one `[SEG]` token for the plume or the source.
- **VQA** (visual question answering): the answer is text only, with no mask.
- **GCG** (grounded conversation generation): the answer describes the scene and links phrases to masks with `<p>phrase</p>[SEG]`.

The model is trained and tested on **GasGround**, a dataset of thermal OGI frames
with plume and source masks and question-answer records for all three levels.

This repository is the code release for the paper "GasVLM: Grounded vision-language
quantification of methane and carbon dioxide in optical gas imaging for livestock
emission monitoring".

## Model

GasVLM has three parts: a frozen
[V-JEPA 2](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) ViT-L image encoder
with a LoRA adapter, the
[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
language model with LoRA, and a multi-scale mask decoder that outputs one mask per
`[SEG]` token. About 58M of its 700M parameters (8.3%) are trainable.

## Results

The scores are the mean ± SD over three training runs with different random seeds
on the GasGround test split. The released checkpoint is one of these runs.

| Level | Metric | Value |
| --- | --- | --- |
| RES | gIoU / cIoU | 70.3 ± 0.2 / 70.1 ± 0.1 |
| RES | gas IoU / source IoU | 69.2 ± 0.1 / 82.9 ± 0.6 |
| VQA | ratio-class accuracy / within-1 | 70.1 ± 2.9 / 95.8 ± 1.6 |
| VQA | CH4 / CO2 MAE | 6.38 ± 0.06 / 6.40 ± 0.05 |
| GCG | mIoU / mask AP50 | 69.8 ± 0.1 / 68.8 ± 1.4 |

## Installation

You need Python 3.10 or later and a CUDA GPU.

```bash
git clone https://github.com/toqitahamid/GasVLM.git
cd GasVLM
pip install -r requirements.txt
pip install -e .
```

The V-JEPA 2 encoder and the SmolLM2 language model download from the Hugging Face
Hub on first use.

## Data

GasGround is on the Hugging Face Hub at
[toqi/GasGround](https://huggingface.co/datasets/toqi/GasGround). You get access
after you accept its terms. GasGround is released under CC BY-NC 4.0, so it may
not be used for commercial purposes.

1. Log in to the Hub with `hf auth login`.
2. Download and unpack the dataset into `data/`, where the config looks for it:

   ```bash
   python tools/unpack_gasground.py --repo-id toqi/GasGround --out data
   ```

This writes `data/{train,val,test}.jsonl` and the frames and masks for each split.
To use your own data, see [docs/DATA.md](docs/DATA.md) for the record schema and
the mask convention.

## Weights

The trained checkpoint is on the Hugging Face Hub at
[toqi/GasVLM](https://huggingface.co/toqi/GasVLM).

```bash
python tools/download_weights.py --repo-id toqi/GasVLM --filename gasvlm.pt
```

The file goes to `checkpoints/gasvlm.pt`. The weights are released under
CC BY-NC 4.0, so they may not be used for commercial purposes.

## Evaluation

Run all three levels on the test split with one GPU:

```bash
bash dist_eval.sh eval.py 1 --config configs/gasvlm_headline.yaml \
    --checkpoint checkpoints/gasvlm.pt --output-dir runs/eval --split test --levels 1,2,3
```

Use `--levels 1` for RES, `--levels 2` for VQA, or `--levels 3` for GCG. The second
argument to `dist_eval.sh` is the number of GPUs.

## Training

```bash
torchrun --standalone --nproc_per_node=1 train.py --config configs/gasvlm_headline.yaml
```

All training settings, such as epochs, learning rate, and batch size, are in
`configs/gasvlm_headline.yaml`. A checkpoint is saved after each epoch to
`runs/gasvlm_headline/epoch_<N>/gasvlm.pt`. The final-epoch checkpoint is used for
evaluation.

Two options are off by default:

- Set `training.seed` in the config to fix the random seed.
- Pass `--val-loss-every-epoch` to log the validation loss after each epoch to `val_curve.jsonl`.

## Repository layout

```
GasVLM/
├── configs/gasvlm_headline.yaml   # model, data, and training settings
├── train.py                       # training script
├── eval.py                        # evaluation script for RES, VQA, and GCG
├── dist_eval.sh                   # torchrun launcher for eval.py
├── gasvlm/
│   ├── models/                    # encoder, projector, language model, mask decoder
│   ├── data/                      # dataset loader and thermal augmentation
│   └── utils/                     # distributed helpers and metrics
├── tools/
│   ├── download_weights.py        # fetch the checkpoint from the Hub
│   └── unpack_gasground.py        # unpack the GasGround shards
├── data_generation/               # code that generated the GasGround question-answer records
└── docs/DATA.md                   # data format and mask convention
```

## Dataset generation

The question-answer records of GasGround were generated with a language model from
the frame labels. The prompts, validator rules, and settings are in
[data_generation/README.md](data_generation/README.md).

## Citation

The paper is under review. A BibTeX entry will be added here after publication.

## License

- Code: MIT. See [LICENSE](LICENSE).
- GasGround dataset and trained weights:
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use is not permitted.

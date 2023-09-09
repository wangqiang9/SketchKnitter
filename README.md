# SketchKnitter
![Python 3.6](https://img.shields.io/badge/python-3.6-green) ![Pytorch 1.6](https://img.shields.io/badge/pytorch-1.6-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green) [![open issues](https://isitmaintained.com/badge/open/xduwq/sketchknitter.svg)](https://github.com/XDUWQ/SketchKnitter/issues)

In this repository, you can find the PyTorch implementation of [SketchKnitter: Vectorized Sketch Generation with Diffusion Models](https://openreview.net/pdf?id=4eJ43EN2g6l), ICLR 2023, Spotlight.

_Authors_: [Qiang Wang](https://scholar.google.com/citations?user=lXyi3t4AAAAJ&hl=en), [Haoge Deng](https://github.com/Bitterdhg), [Yonggang Qi](https://qugank.github.io/), [Da Li](https://scholar.google.co.uk/citations?user=RPvaE3oAAAAJ&hl=en), [Yi-Zhe Song](https://scholar.google.co.uk/citations?hl=en&user=irZFP_AAAAAJ&view_op=list_works&sortby=pubdate). Beijing University of Posts and Telecommunications, Samsung AI Centre Cambridge, University of Surrey.

> Abstract: We show vectorized sketch generation can be identified as a reversal of the stroke deformation process. This relationship was established by means of a diffusion model that learns data distributions over the stroke-point locations and pen states of real human sketches. Given randomly scattered stroke-points, sketch generation becomes a process of deformation-based denoising, where the generator rectifies positions of stroke points at each timestep to converge at a recognizable sketch. A key innovation was to embed recognizability into the reverse time diffusion process. It was observed that the estimated noise during the reversal process is strongly correlated with sketch classification accuracy. An auxiliary recurrent neural network (RNN) was consequently used to quantify recognizability during data sampling. It follows that, based on the recognizability scores, a sampling shortcut function can also be devised that renders better quality sketches with fewer sampling steps. Finally it is shown that the model can be easily extended to a conditional generation framework, where given incomplete and unfaithful sketches, it yields one that is more visually appealing and with higher recognizability.

![Fig.1](https://github.com/XDUWQ/SketchKnitter/blob/main/images/overview.png)

## Datasets
Please go to the [QuickDraw official website](https://quickdraw.withgoogle.com/) to download the datasets. The class list used in the paper: `moon`, `airplane`, `fish`, `umbrella`, `train`, `spider`, `shoe`, `apple`, `lion`, `bus`, you can also replace it with any other category. Each category class is stored in its own file, and contains training/validation/test set sizes of 70000/2500/2500 examples.

In addition to the QuickDraw dataset, you can train the model on any dataset, but please pay attention to organizing the dataset into vector format and packaging it into `.npz` file. In the case of less data sets, please pay attention to over-fitting. If you want to create your own dataset, you can follow the [official tutorial](https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn#creating-your-own-dataset) of SketchRNN.

## Installation
The requirements of this repo can be found in [requirements.txt](https://github.com/XDUWQ/SketchKnitter/blob/main/requirements.txt).
```bash
conda create -n sketchknitter python=3.7
conda activate sektchknitter
pip install -r requirements.txt
```

## Train and Inference

### Haperparameters
Here is a list of full options for the model:
```bash
lr,                   # learning rate.
log_dir,              # save log path.
dropout,              # dropout rate.
use_fp16,             # whether to use mixed precision training.
ema_rate,             # comma-separated list of EMA values
category,             # list of category name to be trained.
data_dir,             # the data sets path.
use_ddim,             # choose whether to use DDIM or DDPM
save_path,            # path to save vector results.
pen_break,            # determines the experience value of stroke break.
image_size,           # the max numbers of datasets.
model_path,           # path to save the trained model checkpoint.
class_cond,           # whether to use guidance technology.
batch_size,           # batch size of training.
emb_channels,         # Unet embedding channel numbers.
num_channels,         # the numbers of channels in Unet backbone.
out_channels,         # output channels in Unet. 
save_interval,        # saving models interval.
noise_schedule,       # the method of adding noise is linear by default.
num_res_blocks,       # numbers of resnet blocks in Unet backbone.
diffusion_steps,      # diffusion steps in the forward process.
schedule_sampler,     # the schedule of sampler.
fp16_scale_growth,    # the mixed precision scale growth.
use_scale_shift_norm, # whether to use scale shift norm. 
```

### Train Example Usage:
```bash
bash train.sh
```

### Inference Example Usage:
```bash
bash sample.sh
```

## Visualization and Evaluation
`sample.py` obtained results are stored in `.npz` file format, which can be directly used to calculate quantitative indicators such as `FID`, `IS`, `GS`, etc. You can also visualize the results for qualitative experiments, refer to the results of Google Brain for the [visualization method](https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn).

To calculate of `FID` and `IS`, refer to the [official code](https://github.com/topics/fid-score). The calculation of [Geometry Score(`GS`)](https://github.com/KhrulkovV/geometry-score) can directly use data in vector format, please go to the [official website](https://github.com/KhrulkovV/geometry-score) for instructions.

## Results
| **Simple** | FID↓ | GS↓ |  Prec↑ | Rec↑ |
| :----:| :----: | :----: | :----: | :----: |
| [SketchPix2seq](https://github.com/MarkMoHR/sketch-pix2seq) | 13.3 | 7.0 | 0.40 | 0.79 |
| [SketchHealer](https://github.com/sgybupt/SketchHealer) | 10.3 | 5.9 | 0.45 | 0.81 |
| [SketchRNN](https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn) | 10.8 | 5.4 | 0.44 | 0.82 |
| [Diff-HW](https://github.com/tcl9876/Diffusion-Handwriting-Generation) | 13.3 | 6.8 | 0.42 | 0.81 |
| [SketchODE](https://github.com/dasayan05/sketchode) | 11.5 | 9.4 | 0.48 | 0.74 |
| Ours (full 1000 steps)  |  **6.9** | **3.4** | **0.52** | **0.88** |
| Ours (r-Shortcut, S=30)  | 7.4 | 3.9 | 0.47 | 0.87 |
| Ours (Linear-DDIMs, S=30) | 11.9 | 6.4 | 0.38 | 0.81 |
| Ours (Quadratic-DDIMs, S=30)  | 12.3 | 6.6 | 0.41 | 0.79 |
| Ours (Abs)  | 20.7 | 12.1 | 0.18 | 0.55 |
| Ours ([Point-Shuffle](https://github.com/qugank/sketch-lattice.github.io)) | 9.5 | 5.3 | 0.35 | 0.72 |
| Ours ([Stroke-Shuffle](https://github.com/qugank/sketch-lattice.github.io)) |  8.2 | 3.8  | 0.36 | 0.74 |

![Fig 4](https://github.com/XDUWQ/SketchKnitter/blob/main/images/4.png)

Only part of the results are listed here. For more detailed results, please see [our paper and supplementary materials](https://openreview.net/pdf?id=4eJ43EN2g6l).

## License
This project is released under the [MIT License](https://github.com/XDUWQ/SketchKnitter/blob/main/LICENSE).

## Citation
If you find this repository useful for your research, please use the following.
```
@inproceedings{wangsketchknitter,
  title={SketchKnitter: Vectorized Sketch Generation with Diffusion Models},
  author={Wang, Qiang and Deng, Haoge and Qi, Yonggang and Li, Da and Song, Yi-Zhe},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```

## Acknowledgements
* [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)
* [guided-diffusion](https://github.com/openai/guided-diffusion)
* [Sketch-R2CNN](https://github.com/craigleili/Sketch-R2CNN)
* [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
* [SketchLattice](https://github.com/qugank/sketch-lattice.github.io)

## Contact
If you have any questions about the code, please contact wanqqiang@bupt.edu.cn 

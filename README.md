# SketchKnitter
![Python 3.6](https://img.shields.io/badge/python-3.6-green) ![Pytorch 1.13](https://img.shields.io/badge/pytorch-1.13-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green)

In this repository, you can find the PyTorch implementation of [SketchKnitter: Vectorized Sketch Generation with Diffusion Models](https://openreview.net/pdf?id=4eJ43EN2g6l). 

_Authors_: [Qiang Wang](https://scholar.google.com/citations?user=lXyi3t4AAAAJ&hl=en), [Haoge Deng](https://github.com/Bitterdhg), [Yonggang Qi](https://qugank.github.io/), [Da Li](https://scholar.google.co.uk/citations?user=RPvaE3oAAAAJ&hl=en), [Yi-Zhe Song](https://scholar.google.co.uk/citations?hl=en&user=irZFP_AAAAAJ&view_op=list_works&sortby=pubdate). Beijing University of Posts and Telecommunications, Samsung AI Centre Cambridge, University of Surrey.

> We show vectorized sketch generation can be identified as a reversal of the stroke deformation process. This relationship was established by means of a diffusion model that learns data distributions over the stroke-point locations and pen states of real human sketches. Given randomly scattered stroke-points, sketch generation becomes a process of deformation-based denoising, where the generator rectifies positions of stroke points at each timestep to converge at a recognizable sketch. A key innovation was to embed recognizability into the reverse time diffusion process. It was observed that the estimated noise during the reversal process is strongly correlated with sketch classification accuracy. An auxiliary recurrent neural network (RNN) was consequently used to quantify recognizability during data sampling. It follows that, based on the recognizability scores, a sampling shortcut function can also be devised that renders better quality sketches with fewer sampling steps. Finally it is shown that the model can be easily extended to a conditional generation framework, where given incomplete and unfaithful sketches, it yields one that is more visually appealing and with higher recognizability.

![Fig.1](https://github.com/XDUWQ/SketchKnitter/blob/main/images/overview.png)

## Datasets
Please go to the [QuickDraw official website](https://github.com/googlecreativelab/quickdraw-dataset) to download the datasets. The class list used in the paper: `moon`, `airplane`, `fish`, `umbrella`, `train`, `spider`, `shoe`, `apple`, `lion`, `bus`, you can also replace it with any other category.

## Installation
The requirements of this repo can be found in [requirements.txt](https://github.com/XDUWQ/DiffSketching/blob/main/requirements.txt).
```
pip install -r requirements.txt
```

## Train


## Test



## Evaluation
Please package the results to be evaluated in `.npz` format, and provide `FID`, `IS`, `Precision` and `Recall` test results.
```
python evaluations/evaluator.py [/path/to/reference-data] [/path/to/generate-data]
```


## Citation
The paper has been accepted by ICLR 2023, and the citation will be released later. 

## Acknowledgements
* [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)
* [guided-diffusion](https://github.com/openai/guided-diffusion)
* [Sketch-R2CNN](https://github.com/craigleili/Sketch-R2CNN)


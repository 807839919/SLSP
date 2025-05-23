# Secrets Lie in Smooth Patches: Synthetic Image Detection with Generator-Agnostic Pixel Fluctuations

An implementation code for paper "Secrets Lie in Smooth Patches: Synthetic Image Detection with Generator-Agnostic Pixel Fluctuations"


## Table of Contents

- [Background](#background)
- [Datasets](#datasets)
- [Dependency](#dependency)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)


## Background

With the proliferation of powerful image generative models, accurately and robustly detecting synthetic images has become an increasingly challenging and important problem. Recent detection approaches often overfit to specific generators or dataset artifacts, limiting their generalizability. Our work revisits the fundamental statistical differences between real and generated images, focusing on the subtle stochastic pixel fluctuations that are inherently present in real photographs but largely missing or over-smoothed in synthetic ones.

We propose Training-Free Skeptical Over-Smooth Region (TrSOR) Selection—a principled, generator-agnostic patch selection algorithm that identifies highly smooth, fluctuation-sensitive regions in an image via a statistical thresholding procedure based on local gradient energy. We then design a lightweight self-supervised pipeline to encode and reconstruct these patches, enabling generator-agnostic synthetic image detection without the need for any fake images during training.

<p align='center'>  
  <img src='Figure/model.jpg' width='400'/>
</p>

<p align='center'>  
  <em>An overview of our architecture. Top: the first step to select Skeptical Over-Smooth Region by iterative estimation. Bottom: thesecond step to self-encode the fluctuations viaa bottleneck paradigm fashion.
</em>
</p>

## Datasets
<p align='center'>  
  <img src='Figure/data.jpg' width='600'/>
</p>



<p align='center'>  
  <em>Some examples of generated images in our Benchmark.
</em>
</p>


## Dependency

### Environment Setup
We recommend using Anaconda to set up the environment:

``` bash
# create conda environment
conda create -n SLSP -y python=3.9
conda activate SLSP

# install dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116  # install torch-1.13.1
pip install accelerate==0.12.0 absl-py ml_collections einops wandb ftfy==6.1.1 transformers==4.23.1
pip install -r requirements.txt

# xformers is optional, but it would greatly speed up the attention computation.
pip install -U xformers
pip install -U --pre triton
```
### Data Preparation
The resulting directory structure should be as follows:

``` bash
data/datasets
├── data
│   ├── train_data
│   │   ├── ImageNet
│   │   ├── COCO
│   │   ├── LSUN
│   │   ├──...
│   ├── test_data
│   │   ├── GauGAN
│   │   ├── StartGAN
│   │   ├──...
├── model
├── ...
```

## Usage

### Training

```bash
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nohup python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 256 \
            --transform_mode 'crop' \
            --model $MODEL \
            --data_path "$train_data" \
            --save_ckpt_freq 5 \
            --batch_size 64 \
            --warmup_epochs 10 \
            --epochs 50 \
            --num_workers 16 \
            --output_dir $OUTPUT_PATH \
```



## Acknowledgments

Our work is developed based on [UViT](https://github.com/baofff/U-ViT), [SAFE](https://github.com/Ouxiang-Li/SAFE/) and [SPAI](https://mever-team.github.io/spai). We sincerely appreciate the efforts of the developers from the previous codebase.


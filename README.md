# Large Language Models are Temporal and Causal Reasoners for Video Question Answering 

This is the official implementation of Flipped-VQA (EMNLP 2023) ([arxiv](https://arxiv.org/abs/2310.15747)) ([demo](https://ikodoh.github.io/flipped_vqa_demo.html)).

> Dohwan Ko<sup>1*</sup>, Ji Soo Lee<sup>1*</sup>, Wooyoung Kang<sup>2</sup>, Byungseok Roh<sup>2</sup>, Hyunwoo J. Kim<sup>1</sup>.
>
><sup>1</sup>Department of Computer Science and Engineering, Korea University   <sup>2</sup>Kakao Brain

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-language-models-are-temporal-and-causal/video-question-answering-on-next-qa)](https://paperswithcode.com/sota/video-question-answering-on-next-qa?p=large-language-models-are-temporal-and-causal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-language-models-are-temporal-and-causal/video-question-answering-on-situated)](https://paperswithcode.com/sota/video-question-answering-on-situated?p=large-language-models-are-temporal-and-causal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-language-models-are-temporal-and-causal/video-question-answering-on-dramaqa)](https://paperswithcode.com/sota/video-question-answering-on-dramaqa?p=large-language-models-are-temporal-and-causal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-language-models-are-temporal-and-causal/video-question-answering-on-vlep)](https://paperswithcode.com/sota/video-question-answering-on-vlep?p=large-language-models-are-temporal-and-causal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-language-models-are-temporal-and-causal/video-question-answering-on-tvqa)](https://paperswithcode.com/sota/video-question-answering-on-tvqa?p=large-language-models-are-temporal-and-causal)

<div align="center">
  <img src="asset/main.png" width="900px" />
</div>

## Setup
To install requirements, run:
```
git clone https://github.com/mlvlab/Flipped-VQA.git
cd Flipped-VQA
mkdir pretrained
conda create -n flipped-vqa python=3.8
conda activate flipped-vqa
sh setup.sh
```

## Dataset & LLaMA Preparation

* You can download our preprocessed datasets (NExT-QA, STAR, DramaQA, VLEP and TVQA) in [huggingface](https://huggingface.co/datasets/ikodoh/Flipped-VQA-Data). We also provide the fine-tuned model on each dataset.

```
git lfs install
git clone https://huggingface.co/datasets/ikodoh/Flipped-VQA-Data
mv ./Flipped-VQA-Data/data ./
mv ./Flipped-VQA-Data/checkpoint ./
unzip ./data/tvqa/tvqa_subtitles.zip -d ./data/tvqa
rm -rf Flipped-VQA-Data ./data/tvqa/tvqa_subtitles.zip
```

* You can download original LLaMA at [here](https://github.com/facebookresearch/llama/tree/llama_v1), and put the checkpoint in ```./pretrained```.

```
./pretrained
   └─ llama
       |─ 7B
       |   |─ consolidated.00.pth
       |   └─ params.json
       |─ 13B
       |   :
       |─ 33B
       |   :
       └─ tokenizer.model
```

## Training LLaMA-VQA (LLaMA + Flipped-VQA)

### NExT-QA

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 7B \
--max_seq_len 128 --batch_size 8 --epochs 5 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/nextqa --accum_iter 2 --vaq --qav
```

### STAR

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 7B \
--max_seq_len 128 --batch_size 8 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star \
--blr 9e-2 --weight_decay 0.16 --output_dir ./checkpoint/star --accum_iter 1 --vaq --qav
```

### DramaQA

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 7B \
--max_seq_len 384 --batch_size 2 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset dramaqa \
--blr 9e-2 --weight_decay 0.10 --output_dir ./checkpoint/dramaqa --accum_iter 8 --vaq --qav
```

### VLEP

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 7B \
--max_seq_len 256 --batch_size 4 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset vlep \
--blr 6e-2 --weight_decay 0.20 --output_dir ./checkpoint/vlep --accum_iter 8 --sub --qav
```

### TVQA

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py --model 7B \
--max_seq_len 650 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset tvqa \
--blr 7e-2 --weight_decay 0.02 --output_dir ./checkpoint/tvqa --dataset tvqa --accum_iter 4 --sub --vaq --qav
```

The fine-tuned checkpoints on each dataset are [here](https://huggingface.co/datasets/ikodoh/Flipped-VQA).

## Evaluation
From the training command, simply replace ```train.py``` with ```eval.py``` and add ```--resume ./your/checkpoint.pth```.

## Acknowledgements

This repo is built upon [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

## Citations

```
@inproceedings{ko2023large,
  title={Large Language Models are Temporal and Causal Reasoners for Video Question Answering},
  author={Ko, Dohwan and Lee, Ji Soo and Kang, Wooyoung and Roh, Byungseok and Kim, Hyunwoo J},
  booktitle={EMNLP},
  year={2023}
}
```


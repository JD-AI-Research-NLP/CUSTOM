# CUSTOM

This repo provides the code for reproducing the experiments in NLPCC2021 paper: [CUSTOM: Aspect-Oriented Product Summarization for E-Commerce](https://arxiv.org/abs/2108.08010)

![Image text](https://github.com/JD-AI-Research-NLP/CUSTOM/blob/main/Images/model_final.jpg)


## Environment
-------
```
tensorflow==1.14.0
torch==0.4.0
```
## PGNet
- Train
    1. ```sh train.sh```
- Predict
    1. ```sh prediction.sh```


## UniLM
- Train
    1. make pretrained_model folder, download Bert-base chinese version and move to this folder
    2. ```sh run_seq2seq.sh```
- Predict
    1. ```sh run_decode.sh```

## Reference
------------
Thanks for your citation:
```
@article{Liang2021CUSTOMAP,
  title={CUSTOM: Aspect-Oriented Product Summarization for E-Commerce},
  author={Jiahui Liang and Junwei Bao and Yifan Wang and Youzheng Wu and Xiaodong He and Bowen Zhou},
  journal={ArXiv},
  year={2021},
  volume={abs/2108.08010}
}
```
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
- 训练
    1. 解压data文件夹
    2. 模型训练
        - 修改train.sh的root路径与PGNet路径一致
        - 修改domain，shouji代表训练数据集，shouji_dev代表验证集，shouji_testFinale代表测试集，diannao数据集同理
        - 对于训练和验证集分别运行打开train_step6和prediction_step6进行数据集预处理
        - 打开train_step8进行模型训练
- 评测
    1. 模型测试
        - 修改prediction.sh的root路径与PGNet路径一致
        - 修改dataset为testFinal
        - 打开prediction_step6进行数据集预处理
        - 打开prediction_step8进行预测


## UniLM模型
- 训练
    1. 下载Bert-base chinese版本解压至pretrained_model文件夹
    2. 运行run_seq2seq.sh，data_dir为指定数据集目录，output_dir和log_dir为模型保存和log路径
- 评测
    1. 运行run_decode.sh，STEP为要评测的模型编号，MODEL_PATH为模型保存路径，output_file为生成结果保存路径

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
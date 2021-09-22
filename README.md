# CUSTOM

This repo provides the code for reproducing the experiments in NLPCC2021 paper: [CUSTOM: Aspect-Oriented Product Summarization for E-Commerce](https://arxiv.org/abs/2108.08010)

![Image text](https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E5%9B%BE%E7%89%87&step_word=&hs=0&pn=2&spn=0&di=102340&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=0&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=undefined&cs=3079226347%2C3026670020&os=3917366788%2C64527412&simid=0%2C0&adpicid=0&lpn=0&ln=675&fr=&fmq=1632300809108_R&fm=&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=11&oriquery=&objurl=https%3A%2F%2Fgimg2.baidu.com%2Fimage_search%2Fsrc%3Dhttp%3A%2F%2Finews.gtimg.com%2Fnewsapp_match%2F0%2F1800232653%2F0%26refer%3Dhttp%3A%2F%2Finews.gtimg.com%26app%3D2002%26size%3Df9999%2C10000%26q%3Da80%26n%3D0%26g%3D0n%26fmt%3Djpeg%3Fsec%3D1634892820%26t%3D3724b9d72882171583565047c7f67345&fromurl=ippr_z2C%24qAzdH3FAzdH3Fetjo_z%26e3Btgjof_z%26e3Bqq_z%26e3Bv54AzdH3FwAzdH3Fdad8aldaAa9NLbaa&gsm=3&rpstart=0&rpnum=0&islist=&querylist=&nojc=undefined)


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
# CUSTOM
- ### PGNet模型运行步骤
    1. 解压data文件夹
    2. 模型训练
        - 修改train.sh的root路径与PGNet路径一致
        - 修改domain，shouji代表训练数据集，shouji_dev代表验证集，shouji_testFinale代表测试集，diannao数据集同理
        - 对于训练和验证集分别运行打开train_step6和prediction_step6进行数据集预处理
        - 打开train_step8进行模型训练

    3. 模型测试
        - 修改prediction.sh的root路径与PGNet路径一致
        - 修改dataset为testFinal
        - 打开prediction_step6进行数据集预处理
        - 打开prediction_step8进行预测


- ### UniLM模型运行步骤

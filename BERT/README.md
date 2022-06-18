基于Pytorch的Bert中文文本情感分类
Pytorch版本 1.4
python 3.7
条件允许可以使用GPU训练
GPU服务器：Tesla P100 16GB 

运行main.py启动程序

- bert_pretrain: BERT预训练模型
- data: 数据集
- models: BERT模型的定义及相关配置
- pytorch的预训练模块

模型下载
1. 下载bert：

下载地址：[GitHub - google-research/bert: TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)

2. 下载bert预训练模型：

pytorch_model.bin
bert_config.json
vocab.txt 

下载地址： [bert-base](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)

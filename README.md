项目包含4个子目录
- Dict
- ML 
- DL
- BERT
  
分别实现基于情感词典、SVM、LSTM和BERT的文本情感分类。

运行每个子项目的main.py文件启动各项目。

环境
- Python 3.7.10
- PyTorch 1.9.0 1.4.0
- Jieba 0.42.1
- Scipy 1.3.1
- Sklearn 0.21.3
- Genism 4.1.2

数据集：[weibo_senti_100k](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k)

由于Github文件大小的限制，部分大文件通过百度网盘上传，包括
- ./BERT/bert_pretrain/pytorch_model.bin		BERT预训练模型
- ./BERT/data/saved_dict/BERT.ckpt			BERT微调后的训练得到的模型
- ./BERT/data/saved_dict/BERT-unfinetune		BERT非微调的训练得到的模型
- ./ML/wiki100d/wiki.zh.text.vector			Wiki100维词向量
下载后放入项目对应位置。
[链接](https://pan.baidu.com/s/1AZLHnoqzZrZpDA72oPolMA?pwd=2034) 
提取码：2034

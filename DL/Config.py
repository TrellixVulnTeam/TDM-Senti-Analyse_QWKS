#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 14:08
# @Author  : Lee
# @FileName: Config.py
# @Email: lishoubo21@mails.ucas.ac.cn

""" 参数配置 """
'''
这里为了使用预训练的中文维基词向量，必须将embedding层的维度设置为50维以和预训练权重匹配
其他的参数如dropout 概率，层数等都可以自定义
'''
class Config(object):
    # 私有变量是map
    # 设置变量的时候，初始化设置map
    def __init__(self, mp):
        self.map = mp
    # set 可以省略，如果直接初始化设置
    def __setattr__(self, name, value):
        if name == 'map':   # 初始化的设置，走默认的方法
            object.__setattr__(self, name, value)
            return
        self.map[name] = value
    # 之所以自己新建一个类就是为了能够实现直接调用名字的功能
    def __getattr__(self, name):
        return self.map[name]

Config = Config({
    'train_path' : '../data/weibo_senti_100k/train_seg.txt',  # 训练集的路径
    'validation_path' : '../data/weibo_senti_100k/dev_seg.txt', # 验证集的路径
    'test_path' : '../data/weibo_senti_100k/test_seg.txt',    # 测试集的路径
    'pred_word2vec_path' : './data/wiki_word2vec_50.bin', # wiki的word2vec词向量的路径
    'tensorboard_path' : './tensorboard',   # 保存训练tensorboard日志文件
    'model_save_path' : './models/senti-classification-model-seg-e5-b16.pth', # 保存模型的路径
    'embedding_dim' : 50,   # 词向量的维度为50维
    'hidden_dim' : 100, # LSTM的隐藏层输出的维度为100维
    'lr' : 0.001,   # 初始学习率为0.001
    'LSTM_layers' : 3,  # LSTM的隐藏层数为3层
    'batch_size' : 16,  # batch_size设置为16
    'Epochs' : 5,  # epoch设置为5
    'num_workers' : 0,   # 数据加载线程数为0
    'drop_prob' : 0.5,  # 神经元dropout的概率为0.5
    'seed' : 0, # 随机种子为0，便于复现模型结构
})
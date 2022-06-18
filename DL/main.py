#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 13:54
# @Author  : Lee
# @FileName: main.py
# @Email: lishoubo21@mails.ucas.ac.cn
import torch
import os
import jieba #分词
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch import nn
import torch.optim as optim
# from tqdm import tqdm
from tqdm import tqdm
from Config import Config
# from Dataset import word2ix, ix2word, train_loader, validation_loader, test_loader
from Dataset import word2ix
from Model import SentimentModel
import train
import test
from set_seed import set_seed
import pre_weight
from EvaluateIndex import ConfuseMeter
from predict import predict
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    set_seed(Config.seed)  # 设置随机种子
    """ 模型初始化 """
    model = SentimentModel(embedding_dim=Config.embedding_dim,
                           hidden_dim=Config.hidden_dim,
                           pre_weight=pre_weight.pre_weight(len(word2ix)))
    print('Model:', model)  # 展示模型网络结构
    """ 模型训练 """
    train.train(model)    # 无需训练时可注释
    """ 测试集相关指标 """
    # 包括精确率，召回率，F1 Score以及混淆矩阵，测试集准确率达到85%,精确率88%，召回率80.7%，F1分数：0.84
    test.test(model)     # 无需测试时可注释
    """ 模型加载 """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(Config.model_save_path), strict=True)  # 模型加载

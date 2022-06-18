#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 21:59
# @Author  : Lee
# @FileName: test.py
# @Email: lishoubo21@mails.ucas.ac.cn
import os
from torch.utils.tensorboard import SummaryWriter
from EvaluateIndex import AvgrageMeter, ConfuseMeter, accuracy
from Model import SentimentModel
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from Config import Config
import pre_weight
from Dataset import test_loader, word2ix

"""
测试函数
"""
def test_helper(validate_loader, device, model, criterion):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    confuse_meter = ConfuseMeter()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = tqdm(validate_loader)
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
            #         inputs,labels = data[0],data[1]
            outputs,_ = model(inputs, batch_seq_len)
#             loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            confuse_meter.update(outputs, labels)
#             validate_loss += loss.item()
            postfix = { 'test_acc': '%.6f' % val_top1.avg,
                      'confuse_acc': '%.6f' % confuse_meter.acc}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return confuse_meter

def test(model):
    # 包括精确率，召回率，F1 Score以及混淆矩阵，测试集准确率达到85%,精确率88%，召回率80.7%，F1分数：0.84
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    model_test = SentimentModel(embedding_dim=Config.embedding_dim,
                                hidden_dim=Config.hidden_dim,
                                pre_weight=pre_weight.pre_weight(len(word2ix)))
    optimizer_test = optim.Adam(model_test.parameters(), lr=Config.lr)
    scheduler_test = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调整
    criterion_test = nn.CrossEntropyLoss()
    model_test.load_state_dict(torch.load(Config.model_save_path), strict=True)  # 模型加载
    confuse_meter = ConfuseMeter()
    confuse_meter = test_helper(test_loader, device, model_test, criterion_test)
    print('prec:%.6f  recall:%.6f  F1:%.6f' % (confuse_meter.pre, confuse_meter.rec, confuse_meter.F1))
    # 混淆矩阵
    print('混淆矩阵:', confuse_meter.confuse_mat)
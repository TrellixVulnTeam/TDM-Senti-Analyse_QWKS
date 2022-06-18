#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 21:52
# @Author  : Lee
# @FileName: train.py
# @Email: lishoubo21@mails.ucas.ac.cn
"""
训练函数
"""
import torch
import os
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from EvaluateIndex import AvgrageMeter, ConfuseMeter, accuracy
from Config import Config
from tqdm import tqdm
from validate import validate
from Dataset import train_loader, test_loader, validation_loader
import pre_weight

#一个epoch的训练逻辑
def train_helper(epoch,epochs, train_loader, device, model, criterion, optimizer,scheduler,tensorboard_path):
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
        inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
        # 初始为0，清除上个batch的梯度信息
        optimizer.zero_grad()
        outputs,hidden = model(inputs,batch_seq_len)

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        _,pred = outputs.topk(1)
        prec1, prec2= accuracy(outputs, labels, topk=(1,2))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
        train_loader.set_postfix(log=postfix)

        # ternsorboard 曲线绘制
        # if os.path.exists(tensorboard_path) == False:
        #     os.mkdir(tensorboard_path)
        # writer = SummaryWriter(tensorboard_path)
        # writer.add_scalar('Train/Loss', loss.item(), epoch)
        # writer.add_scalar('Train/Accuracy', top1.avg, epoch)
        # writer.flush()
    scheduler.step()

def train(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 3
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # 学习率调整
    criterion = nn.CrossEntropyLoss()

    """ 迭代训练 """
    # 在每个epoch中同时收集验证集准确率，防止过拟合
    import shutil
    if os.path.exists(Config.tensorboard_path):
        shutil.rmtree(Config.tensorboard_path)
        os.mkdir(Config.tensorboard_path)

    epochs = Config.Epochs
    for epoch in range(epochs):
        train_loader_tqdm = tqdm(train_loader)
        train_loader_tqdm.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
        train_helper(epoch, epochs, train_loader_tqdm, device, model, criterion, optimizer, scheduler, Config.tensorboard_path)
        validate(epoch, validation_loader, device, model, criterion, Config.tensorboard_path)

    """ 模型保存 """
    if not os.path.exists('./models/'):
        os.mkdir('./models/')
    torch.save(model.state_dict(), Config.model_save_path)

#     print('Finished Training')
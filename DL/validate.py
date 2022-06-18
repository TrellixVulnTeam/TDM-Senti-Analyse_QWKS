#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 21:56
# @Author  : Lee
# @FileName: validate.py
# @Email: lishoubo21@mails.ucas.ac.cn
import os
from torch.utils.tensorboard import SummaryWriter
from EvaluateIndex import AvgrageMeter, ConfuseMeter, accuracy
import torch
from tqdm import tqdm

"""
验证函数
"""
def validate(epoch, validate_loader, device, model, criterion, tensorboard_path):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = tqdm(validate_loader)
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
            #         inputs,labels = data[0],data[1]
            outputs, _ = model(inputs, batch_seq_len)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validate_loss': '%.6f' % (validate_loss / (i + 1)), 'validate_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)

            # ternsorboard 曲线绘制
            # if os.path.exists(tensorboard_path) == False:
            #     os.mkdir(tensorboard_path)
            # writer = SummaryWriter(tensorboard_path)
            # writer.add_scalar('Validate/Loss', loss.item(), epoch)
            # writer.add_scalar('Validate/Accuracy', val_top1.avg, epoch)
            # writer.flush()
        val_acc = val_top1.avg
    return val_acc

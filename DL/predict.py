#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 22:26
# @Author  : Lee
# @FileName: predict.py
# @Email: lishoubo21@mails.ucas.ac.cn
import jieba
import torch
from Dataset import word2ix

def predict(comment_str, model, device):
    model = model.to(device)
    seg_list = jieba.lcut(comment_str,cut_all=False)
    words_to_idx = []
    for w in seg_list:
        try:
            index = word2ix[w]
        except:
            index = 0 #可能出现没有收录的词语，置为0
        words_to_idx.append(index)
    inputs = torch.tensor(words_to_idx).to(device)
    inputs = inputs.reshape(1,len(inputs))
    outputs,_ = model(inputs, [len(inputs),])
    pred = outputs.argmax(1).item()
    return pred
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 22:06
# @Author  : Lee
# @FileName: set_seed.py
# @Email: lishoubo21@mails.ucas.ac.cn
import torch
import numpy as np
import random

# 随机种子设置
'''
随机种子的设置需要在模型初始之前，这样才能保证模型每次初始化的时候得到的是一样的权重，
从而保证能够复现每次训练结果 torch.backends.cudnn.benchmark = True
'''
def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 并行gpu
        torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
#         torch.backends.cudnn.benchmark = True   # 训练集变化不大时使训练加速
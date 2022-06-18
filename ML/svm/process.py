#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 23:32
# @Author  : Lee
# @FileName: process.py
# @Email: lishoubo21@mails.ucas.ac.cn
import pandas as pd

def process():
    csv_data = pd.read_csv('../../data/weibo_senti_100k/test.csv', sep='\t')
    print(csv_data.head())

    csv_data.loc[csv_data['label'] == 0, 'x_test'].to_csv('../tmp/weibo_neg.txt', columns=None, index=False)
    csv_data.loc[csv_data['label'] == 1, 'x_test'].to_csv('../tmp/weibo_pos.txt', columns=None, index=False)

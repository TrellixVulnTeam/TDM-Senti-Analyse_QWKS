#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 15:47
# @Author  : Lee
# @FileName: main.py
# @Email: lishoubo21@mails.ucas.ac.cn
import process
import segmentation
import delStopwords
import word2vec
import pca_svm

if __name__ == '__main__':
    process.process() # 数据集分割
    segmentation.segmentation()   # 中文分词
    delStopwords.delStopwords()   # 去停用词
    word2vec.word2vector()  # 训练word2vec词向量
    pca_svm.pca_svm()   # 预测与评估
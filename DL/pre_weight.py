#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 22:09
# @Author  : Lee
# @FileName: pre_weight.py
# @Email: lishoubo21@mails.ucas.ac.cn
import torch
import gensim # word2vec预训练加载
from Dataset import word2ix, ix2word
from Config import Config

## 预训练权重加载
'''
这里需要将预训练的中文word2vec的权重初始到pytorch embedding层，主要的逻辑思路首先使用gensim包来加载权重，
然后根据前面建立的词汇表，初始一个vocab_size*embedding_dim的0矩阵weight，之后对每个词汇查询是否在预训练的word2vec中有权重，
如果有的话就将这个权重复制到weight中，最后使用weight来初始embedding层就可以了。
'''
# word2vec加载
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(Config.pred_word2vec_path, binary=True)
# 50维的向量
print('word2vec预训练权重的shape:', word2vec_model.__dict__['vectors'].shape)

def pre_weight(vocab_size):
    weight = torch.zeros(vocab_size, Config.embedding_dim)
    # 初始权重
    for i in range(len(word2vec_model.index_to_key)):  # 预训练中没有word2ix，所以只能用索引来遍历
        try:
            index = word2ix[word2vec_model.index2word[i]]  # 得到预训练中的词汇的新索引
        except:
            continue
        weight[index, :] = torch.from_numpy(word2vec_model.get_vector(
            ix2word[word2ix[word2vec_model.index2word[i]]]))  # 得到对应的词向量
    return weight
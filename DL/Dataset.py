#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 14:02
# @Author  : Lee
# @FileName: Dataset.py
# @Email: lishoubo21@mails.ucas.ac.cn
from zhconv import convert #简繁转换
import re #split使用
import torch
# 变长序列的处理
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.utils.data import Dataset,DataLoader
from Config import Config

"""
词汇表建立
首先建立训练数据的词汇表，实现汉字转索引。构建词汇表的逻辑：首先读取训练集的数据，
然后使用zhconv包统一转换成简体， 因为数据集本身就已经是分词后的数据了，
只需要对应的读入这些词汇然后去重，之后根据去重的list构建两个word2ix 和ix2word即可。

这里思路比较简单，但是有个坑，导致我调了一天的bug。
就是每次set操作后对应的顺序是不同的，因为我没有将词汇表保存下来，想的是每次程序运行的时候再来重新构建，
因此每次重新set之后得到的词汇表也是不一致的，导致同样的语言文本经过不同的词汇表转换后，每次都得到不同的输入，
因此导致训练好的模型每次重新加载kernel之后得到的测试集准确率都不一样。
"""

# 简繁转换 并构建词汇表
def build_word_dict(train_path):
    words = []
    max_len = 0
    total_len = 0
    with open(train_path,'r',encoding='UTF-8') as f:
        lines = f.readlines()
        for line in  lines:
            line = convert(line, 'zh-cn') #转换成大陆简体
            line_words = re.split(r'[\s]', line)[1:-1] # 按照空字符\t\n 空格来切分
            max_len = max(max_len, len(line_words))
            total_len += len(line_words)
            for w in line_words:
                words.append(w)
    words = list(set(words))#最终去重
    words = sorted(words) # 一定要排序不然每次读取后生成此表都不一致，主要是set后顺序不同
    #用unknown来表示不在训练语料中的词汇
    word2ix = {w:i+1 for i,w in enumerate(words)} # 第0是unknown的 所以i+1
    ix2word = {i+1:w for i,w in enumerate(words)}
    word2ix['<unk>'] = 0
    ix2word[0] = '<unk>'
    avg_len = total_len / len(lines)
    return word2ix, ix2word, max_len,  avg_len

"""
    数据变长处理
    输入样本的中，词汇的长度不一致，最大的长度有679个词，平均而言只有44个词，
    所以如果只是单纯的填0来进行维度统一的话，大量的0填充会让模型产生误差， 
    为了处理这种情况需要将序列长度不一致的样本，根据长度排序后进行按照批次来分别填充。
"""
def mycollate_fn(data):
    # 这里的data是getittem返回的（input，label）的二元组，总共有batch_size个
    data.sort(key=lambda x: len(x[0]), reverse=True)  # 根据input来排序
    data_length = [len(sq[0]) for sq in data]
    input_data = []
    label_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])
    input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
    label_data = torch.tensor(label_data)
    return input_data, label_data, data_length

"""
    数据集的类里面主要是获取数据和标签，稍微需要注意的是考虑到测试集和验证集中一些不会在训练语料库中出现的词汇，
    需要将这些词汇置为0，来避免索引错误
"""
class CommentDataSet(Dataset):
    def __init__(self, data_path, word2ix, ix2word):
        self.data_path = data_path
        self.word2ix = word2ix
        self.ix2word = ix2word
        self.data, self.label = self.get_data_label()

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

    def get_data_label(self):
        data = []
        label = []
        with open(self.data_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    label.append(torch.tensor(int(line[0]), dtype=torch.int64))
                except BaseException:  # 遇到首个字符不是标签的就跳过比如空行，并打印
                    print('not expected line:' + line)
                    continue
                line = convert(line, 'zh-cn')  # 转换成大陆简体
                line_words = re.split(r'[\s]', line)[1:-1]  # 按照空字符\t\n 空格来切分
                words_to_idx = []
                for w in line_words:
                    try:
                        index = self.word2ix[w]
                    except BaseException:
                        index = 0  # 测试集，验证集中可能出现没有收录的词语，置为0
                    #                 words_to_idx = [self.word2ix[w] for w in line_words]
                    words_to_idx.append(index)
                data.append(torch.tensor(words_to_idx, dtype=torch.int64))
        return data, label

## 训练集，验证集，测试集，加载
word2ix, ix2word, max_len, avg_len = build_word_dict(Config.train_path)
print('数据集文本的最大长度和平均长度:' , max_len, avg_len)

train_data = CommentDataSet(Config.train_path, word2ix, ix2word)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                          num_workers=0, collate_fn=mycollate_fn, )

validation_data = CommentDataSet(Config.validation_path, word2ix, ix2word)
validation_loader = DataLoader(validation_data, batch_size=16, shuffle=True,
                               num_workers=0, collate_fn=mycollate_fn, )

test_data = CommentDataSet(Config.test_path, word2ix, ix2word)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False,
                         num_workers=0, collate_fn=mycollate_fn, )
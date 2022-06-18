#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 21:43
# @Author  : Lee
# @FileName: Model.py
# @Email: lishoubo21@mails.ucas.ac.cn
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence # 变长序列的处理
from Config import Config

"""
模型构建:
模型的构建与LSTM大体一致,即embedding后LSTM层然后3层全连接，激活函数选择了tanh。
不同的点在于，这里的输出只保留时间步的最后一步，用来当作预测结果。也就是最后一个全连接层的输出取最后一个时间步的输出。
以及为了防止过拟合而采用了Dropout
"""

class SentimentModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight):
        super(SentimentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embeddings.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True, dropout=Config.drop_prob, bidirectional=False)
        self.dropout = nn.Dropout(Config.drop_prob)
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    #         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, batch_seq_len, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        embeds = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(Config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))  # hidden 是h,和c 这两个隐状态
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.dropout(torch.tanh(self.fc1(output)))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        last_outputs = self.get_last_output(output, batch_seq_len)
        #         output = output.reshape(batch_size * seq_len, -1)
        return last_outputs, hidden

    def get_last_output(self, output, batch_seq_len):
        last_outputs = torch.zeros((output.shape[0], output.shape[2]))
        for i in range(len(batch_seq_len)):
            last_outputs[i] = output[i][batch_seq_len[i] - 1]  # index 是长度 -1
        last_outputs = last_outputs.to(output.device)
        return last_outputs
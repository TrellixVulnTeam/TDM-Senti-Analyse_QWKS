#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 0:16
# @Author  : Lee
# @FileName: word2vec.py
# @Email: lishoubo21@mails.ucas.ac.cn
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim

# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])  # model[word]：获取这个词的词向量
        except KeyError:
            continue
    return np.array(vecs, dtype='float')


# 构建文档词向量
def buildVecs(filename,model, logger):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8',errors='ignore') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)   #返回的是一条评论的词向量
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean 每条句子词向量的平均值 对应相加的平均值
                #print vecsArray
                #sys.exit()
                fileVecs.append(vecsArray)
    return fileVecs


# if __name__ == '__main__':
def word2vector():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)  # 日志格式
    logger.info("running %s" % ' '.join(sys.argv))

    # load word2vec model
    fdir = '../wiki100d/'
    inp = fdir + 'wiki.zh.text.vector'  # 训练好的词向量（使用wiki中文语料训练的）
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)  # 加载词向量

    posInput = buildVecs('../tmp/weibo_pos_cut_stopword.txt', model, logger)
    negInput = buildVecs('../tmp/weibo_neg_cut_stopword.txt', model, logger)

    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))

    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    # write in file
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y, df_x], axis=1)
    # print data
    data.to_csv(fdir + 'weibodata_vector.csv')

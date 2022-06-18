#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 12:28
# @Author  : Lee
# @FileName: data_preprocess.py
# @Email: lishoubo21@mails.ucas.ac.cn
import codecs
import re
import string

import jieba
import pandas as pd

pd_train = pd.read_csv('../data/train_raw.csv', sep='\t')
pd_dev = pd.read_csv('../data/dev_raw.csv', sep='\t')
pd_test = pd.read_csv('../data/test_raw.csv', sep='\t')
print('train评论数目（总体）：%d' % pd_train.shape[0])
print('train评论数目（正向）：%d' % pd_train[pd_train.label==1].shape[0])
print('train评论数目（负向）：%d' % pd_train[pd_train.label==0].shape[0])
print(pd_train.sample(20))

"""文本分词"""
def prepareData(sourceFile,targetFile):
    f = codecs.open(sourceFile,'r',encoding='utf-8',errors='ignore')
    target = codecs.open(targetFile,'w',encoding='utf-8',errors='ignore')
    print ('open dictionary file: '+ sourceFile)
    print ('open target file: '+ targetFile)

    lineNum = 1
    line = f.readline()  #读取一行
#     print(line)
    while(line):
        print('---processing ',lineNum,' article---')
        label = line[0]  # 保存当前文本的标签0、1
        line = clearTxt(line)   #数据清洗
        seg_line = sent2word(line)  #分词
        seg_line = label + '\t' + seg_line  # 拼接上标签
        target.writelines(seg_line+'\n')
        lineNum = lineNum + 1
        line = f.readline()
    print("well done")
    f.close()
    target.close()

#清洗文本
def clearTxt(line):
    if line !='':
        line = line.strip()      #移除字符串头尾指定的字符（默认为空格或换行符）
        print(line)
        intab = ""
        outtab = ""
        pun_num = string.punctuation + string.digits #所有的标点字符 + 数字0-9
        #print(pun_num)
        trantab = str.maketrans(intab, outtab,pun_num)#创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。并删去 pun_num中的字符
    #     line = line.encode('utf-8')
        #print(line)
        line = line.translate(trantab)  #除去 pun_num中的字符  Python3在maketrans
    #     line = line.decode("utf8")
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]","",line)
        #去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]", "",line)
        #print(line)   #清洗之后的文本
        print('has label:' + line)
    return line

#文本切割
def sent2word(line):
    segList = jieba.cut(line,cut_all=False)
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()

def stopWord(sourceFile,targetFile,stopkey):
    sourcef = codecs.open(sourceFile, 'r', encoding='utf-8')
    targetf = codecs.open(targetFile, 'w', encoding='utf-8')
    print ('open dictionary file: '+ sourceFile)
    print ('open target file: '+ targetFile)
    lineNum = 1
    line = sourcef.readline()
    while line:
        print ('---processing ',lineNum,' article---')
        sentence = delstopword(line,stopkey)
        #print sentence
        targetf.writelines(sentence + '\n')
        lineNum = lineNum + 1
        line = sourcef.readline()
    print ('well done.')
    sourcef.close()
    targetf.close()

#删除停用词
def delstopword(line,stopkey):
    wordList = line.split(' ')    #去除句子首尾的空格
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:   #去除停用词
            if word != '\t':
                sentence += word + " "
    return sentence.strip()

if __name__ == '__main__':
    stopkey = [w.strip() for w in codecs.open('data/stopWord.txt', 'r', encoding='utf-8').readlines()]

    # 训练集数据预处理
    sourceFile = '../data/train_raw.csv'
    segFile = 'data/train_seg.txt'
    targetFile = 'data/train.txt'
    prepareData(sourceFile, segFile)  # 中文分词
    stopWord(segFile, targetFile, stopkey)  # 去停用词

    # 验证集数据预处理
    sourceFile = 'data/dev_raw.csv'
    segFile = 'data/dev_seg.txt'
    targetFile = 'data/dev.txt'
    prepareData(sourceFile, segFile)  # 中文分词
    stopWord(segFile, targetFile, stopkey)  # 去停用词

    # 测试集数据预处理
    sourceFile = 'data/test_raw.csv'
    segFile = 'data/test_seg.txt'
    targetFile = 'data/test.txt'
    prepareData(sourceFile, segFile)    # 中文分词
    stopWord(segFile, targetFile, stopkey)   # 去停用词


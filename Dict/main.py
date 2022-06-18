#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 10:43
# @Author  : Lee
# @FileName: dict.py
# @Email: lishoubo21@mails.ucas.ac.cn
from collections import defaultdict
import os
import re
import jieba
import codecs
# 生成stopword表，需要去除一些否定词和程度词汇
from matplotlib import pyplot as plt
from sklearn import metrics

stopwords = set()
fr = open('dictionary/停用词.txt', 'r', encoding='utf-8')
for word in fr:
    stopwords.add(word.strip())  # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
# 读取否定词文件
not_word_file = open('dictionary/否定词.txt', 'r+', encoding='utf-8')
not_word_list = not_word_file.readlines()
not_word_list = [w.strip() for w in not_word_list]
# 读取程度副词文件
degree_file = open('dictionary/程度副词.txt', 'r+', encoding='utf-8')
degree_list = degree_file.readlines()
degree_list = [item.split(',')[0] for item in degree_list]
# 生成新的停用词表
with open('dictionary/stopwords.txt', 'w', encoding='utf-8') as f:
    for word in stopwords:
        if (word not in not_word_list) and (word not in degree_list):
            f.write(word + '\n')


# jieba分词后去除停用词
def seg_word(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    stopwords = set()
    with open('dictionary/stopwords.txt', 'r', encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x: x not in stopwords, seg_result))


# 找出文本中的情感词、否定词和程度副词
def classify_words(word_list):
    # 读取情感词典文件
    sen_file = open('dictionary/BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    # 获取词典文件内容
    sen_list = sen_file.readlines()
    # 创建情感字典
    sen_dict = defaultdict()
    # 读取词典每一行的内容，将其转换成字典对象，key为情感词，value为其对应的权重
    for i in sen_list:
        if len(i.split(' ')) == 2:
            sen_dict[i.split(' ')[0]] = i.split(' ')[1]

    # 读取否定词文件
    not_word_file = open('dictionary/否定词.txt', 'r+', encoding='utf-8')
    not_word_list = not_word_file.readlines()
    # 读取程度副词文件
    degree_file = open('dictionary/程度副词.txt', 'r+', encoding='utf-8')
    degree_list = degree_file.readlines()
    degree_dict = defaultdict()
    for i in degree_list:
        degree_dict[i.split(',')[0]] = i.split(',')[1]

    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    # 分类
    for i in range(len(word_list)):
        word = word_list[i]
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[i] = sen_dict[word]
        elif word in not_word_list and word not in degree_dict.keys():
            # 分词结果中在否定词列表中的词
            not_word[i] = -1
        elif word in degree_dict.keys():
            # 分词结果中在程度副词中的词
            degree_word[i] = degree_dict[word]

    # 关闭打开的文件
    sen_file.close()
    not_word_file.close()
    degree_file.close()
    # 返回分类结果
    return sen_word, not_word, degree_word


# 计算情感词的分数
def score_sentiment(sen_word, not_word, degree_word, seg_result):
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合
    sentiment_index_list = list(sen_word.keys())
    # 遍历分词结果
    for i in range(0, len(seg_result)):
        # 如果是情感词
        if i in sen_word.keys():
            # 权重*情感词得分
            score += W * float(sen_word[i])
            # 情感词下标加一，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                # 判断当前的情感词与下一个情感词之间是否有程度副词或否定词
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    # 更新权重，如果有否定词，权重取反
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        W *= float(degree_word[j])
        # 定位到下一个情感词
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score


# 计算得分
def sentiment_score(sentence):
    # 1.对文档分词
    seg_list = seg_word(sentence)
    # 2.将分词结果转换成字典，找出情感词、否定词和程度副词
    sen_word, not_word, degree_word = classify_words(seg_list)
    # 3.计算得分
    score = score_sentiment(sen_word, not_word, degree_word, seg_list)
    # 4.如果得分为正数，输出为正类
    if (score > 0):
        return 1
    else:
        return 0
    # return score

# 计算测试集准确率
def test(test_file):
    f = codecs.open(test_file, 'r', encoding='utf-8', errors='ignore')
    print('open dictionary file: ' + test_file)
    lineNum = 1 # 计算样本数量
    f.readline()  # 读取第一行，这行是标签，舍去
    line = f.readline()  # 读取下一行
    y_true = []
    y_pred = []
    # 计算TP FP TN FN
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    while (line):
        print('---classify ', lineNum, ' article---')
        label = int(line[0])  # 保存当前文本的标签0、1
        y_true.append(label)    # 加入label列表
        text = line
        # print('line:', line)
        # print('label:', label)
        text_pred = sentiment_score(text)
        y_pred.append(text_pred)
        if (text_pred == label):
            if label == 1:
                TP += 1
            else:
                TN += 1
        else:
            if (label == 1):
                FN += 1
            else:
                FP += 1
        lineNum += 1
        line = f.readline()
    print("well done")
    f.close()
    acc = (TP + TN) / (lineNum - 1)
    print('测试集样本数量为:', lineNum - 1)
    return TP, TN, FP, FN

if __name__ == '__main__':

    # TP, TN, FP, FN = test('./data/test_raw.txt')
    TP, TN, FP, FN = test('../data/weibo_senti_100k/test.csv')

    total = TP + TN + FP + FN
    neg_test = FP + TN
    pos_test = TP + FN
    print('well done!')

    print("TP, FN = ", TP, TN)
    print("FP, TN = ", FP, FN)

    acc = (TP + TN) / (neg_test + pos_test)
    error = (FN + FP) / (neg_test + pos_test)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F_score = 2 * precision * recall / (precision + recall)
    print("准确率：{:.3f}\n错误率：{:.3f}\n精准率：{:.3f}\n召回率：{:.3f}\nF-score：{:.3f}".format(acc, error, precision, recall, F_score))

    # print('情感词典方法的分类准确率为:', accuracy)
    print('一看微博就想摔手机！[怒]    ', sentiment_score('一看微博就想摔手机！[怒]'))
    print('太阳来了！心情舒畅啦！！[哈哈][哈哈]   ', sentiment_score('太阳来了！心情舒畅啦！！[哈哈][哈哈]'))
    print('啥都不说了，各位达人新年快乐！！！！[哈哈][哈哈][哈哈]    ', sentiment_score('啥都不说了，各位达人新年快乐！！！！[哈哈][哈哈][哈哈]'))
    print('我怎忍心看你流下的眼泪，小皮[泪]    ', sentiment_score('我怎忍心看你流下的眼泪，小皮[泪]'))
    print('路过天安门 我都看不清楚毛爷爷了 我伤心我难过[衰][泪]    ', sentiment_score('路过天安门 我都看不清楚毛爷爷了 我伤心我难过[衰][泪]'))



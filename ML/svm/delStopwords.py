#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 0:13
# @Author  : Lee
# @FileName: delStopwords.py
# @Email: lishoubo21@mails.ucas.ac.cn
import codecs,sys

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
        targetf.writelines('\n')
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


# if __name__ == '__main__':
def delStopwords():
    stopkey = [w.strip() for w in codecs.open('../data/stopWord.txt', 'r', encoding='utf-8').readlines()]

    sourceFile = '../tmp/weibo_neg_cut.txt'
    targetFile = '../tmp/weibo_neg_cut_stopword.txt'
    stopWord(sourceFile, targetFile, stopkey)

    sourceFile = '../tmp/weibo_pos_cut.txt'
    targetFile = '../tmp/weibo_pos_cut_stopword.txt'
    stopWord(sourceFile, targetFile, stopkey)
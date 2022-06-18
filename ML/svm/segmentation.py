#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 0:11
# @Author  : Lee
# @FileName: segmentation.py
# @Email: lishoubo21@mails.ucas.ac.cn
import jieba
import jieba.analyse
import codecs,sys,string,re

#文本分词
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
        line = clearTxt(line)   #数据清洗
        seg_line = sent2word(line)  #分词
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
    return line

#文本切割
def sent2word(line):
    segList = jieba.cut(line,cut_all=False)
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()


# if __name__ == '__main__':
def segmentation():
    sourceFile = '../tmp/weibo_neg.txt'
    targetFile = '../tmp/weibo_neg_cut.txt'
    prepareData(sourceFile, targetFile)

    sourceFile = '../tmp/weibo_pos.txt'
    targetFile = '../tmp/weibo_pos_cut.txt'
    prepareData(sourceFile, targetFile)
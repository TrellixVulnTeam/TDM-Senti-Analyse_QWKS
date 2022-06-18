#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 0:22
# @Author  : Lee
# @FileName: pca_svm.py
# @Email: lishoubo21@mails.ucas.ac.cn
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# PCA降维
def Pca_data(x,components):
    ##计算全部贡献率
    n_components = 100
    pca = PCA(n_components=n_components)   #n_components：  PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
    pca.fit(x)
    #print (pca.explained_variance_ratio_)  # pca.explained_variance_ratio_ 每个特征的信息占比 它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。

    ##PCA作图
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)  # explained_variance_，它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()  # 80维就能够较好的包含原始数据的绝大部分内容
    plt.savefig('result/pca.jpg')


    x_pca = PCA(n_components = components).fit_transform(x)   #降到 80 维
    return x_pca

# 划分训练集和测试集
def Train_Test_split(x_pca,y):
    train_X, test_X, train_Y, test_Y = train_test_split(x_pca, y, train_size=0.8)  # 划分训练集和测试集
    return train_X,test_X,train_Y,test_Y

# svm (RBF)
# using training data with 80 dimensions
#训练
def train(train_X,train_Y):
    clf = svm.SVC(C = 2, probability = True)
    clf.fit(train_X,train_Y)

    # save model
    joblib.dump(clf, 'saved_model/clf_weibo.pkl')

    return clf

# 加载模型
def load_model(model_path):
    clf = joblib.load(model_path)  # 加载训练好的svm模型
    return clf

#评估，ROC图
def eval(x_pca, clf, y):
    print ('Test Accuracy: %.2f'% clf.score(x_pca,y))   #评价模型的好坏
    #Create ROC curve
    pred_probas = clf.predict_proba(x_pca)[:,1] #score  返回预测属于某标签的概率
    fpr,tpr,_ = metrics.roc_curve(y, pred_probas)  #这个方法主要用来计算ROC曲线面积的
    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc = 'lower right')
    plt.show()
    plt.savefig('result/roc.jpg')

#分类报告  clf 是训练好的模型
def class_report(clf, test_X, test_Y):
    target_names = ['1', '0']  # 1：积极  0：消极
    print(classification_report(test_Y, clf.predict(test_X), target_names=target_names))  # SVC

def single_sentence(x_pca, clf, y):
    pred_probas = clf.predict_proba(x_pca)[:, 1]  # score  返回预测属于某标签的概率
    print('单句预测属于1类(Positive)的概率:')
    print('一看微博就想摔手机！[怒]    ', pred_probas[1008])
    print('太阳来了！心情舒畅啦！！[哈哈][哈哈]    ', pred_probas[1948])
    print('啥都不说了，各位达人新年快乐！！！！[哈哈][哈哈][哈哈]   ', pred_probas[3577])
    print('我怎忍心看你流下的眼泪，小皮[泪]    ', pred_probas[8073])
    print('路过天安门 我都看不清楚毛爷爷了 我伤心我难过[衰][泪]    ', pred_probas[10727])

def pca_svm():
    # 获取数据 [95615 rows x 102 columns]
    fdir = '../wiki100d/'
    df = pd.read_csv(fdir + 'weibodata_vector.csv')
    print(df.shape)
    y = df.iloc[:, 1]  # 标签
    x = df.iloc[:, 2:]  # 数值
    # print(x)
    x_pca = Pca_data(x,80)  #降维到 80 维
    train_X, test_X, train_Y, test_Y = Train_Test_split(x_pca,y)    #划分训练集和测试集

    #保存模型的操作在train方法里，
    # clf = train(x_pca,y)   #训练
    clf = load_model('saved_model/clf_weibo.pkl')   # 模型载入

    # single_sentence(x_pca, clf, y)  # 单句预测

    eval(x_pca, clf, y)  #ROC图

    #打印分类报告
    class_report(clf, test_X, test_Y)
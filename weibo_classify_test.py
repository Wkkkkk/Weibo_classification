# coding: utf-8
import io
import jieba
import numpy
import random
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import re
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def string_preprocess(string):
    raw_string = string
    http_info = re.compile('[a-zA-z]+://[^\s]*')
    string_without_http = http_info.sub(ur'链接', raw_string)
    at_info = re.compile(ur'@[^ @，,。.]*')
    string_without_http_and_at = at_info.sub(ur'@', string_without_http)
    number_eng_info = re.compile(ur'[0-9|a-zA-Z]')
    clean_string = number_eng_info.sub('', string_without_http_and_at)
    return clean_string


def input_data(file_name=ur'data\8000条测试数据.xlsx', threshold=5000, seed=25):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    filename = file_name
    df = pd.read_excel(filename, sheetname=2, index_col=None, header=None)
    rdata = df[:threshold]

    random_list = range(threshold)
    random.seed(seed)
    random_index = random.sample(random_list, threshold/5)

    for i in rdata.values:
        cur_tag = i[1]
        cur_string = i[2]
        cur_string = string_preprocess(cur_string)
        if i[0] in random_index:
            test_words.append(cur_string)
            test_tags.append(cur_tag)
        else:
            train_words.append(cur_string)
            train_tags.append(cur_tag)
    # 从文件导入停用词表
    with io.open(ur"data\stop_words.txt", 'r', encoding='utf-8') as f:
        stpwrd_content = f.read()
        stpwrdlst = stpwrd_content.splitlines()

    return train_words, train_tags, test_words, test_tags, stpwrdlst


def vectorize(train_words, test_words, stop_words):

    # v = HashingVectorizer(tokenizer=lambda x: jieba.cut(x, cut_all=True), n_features=30000, non_negative=True,
    #                       stop_words=stop_words)
    v = TfidfVectorizer(tokenizer=lambda x: jieba.cut(x), analyzer='word', stop_words=stop_words)

    train_data = v.fit_transform(train_words)
    test_data = v.transform(test_words)

    return train_data, test_data


def evaluate(actual, pred, pos=1):
    m_precision = metrics.precision_score(actual, pred, pos_label=pos)
    m_recall = metrics.recall_score(actual, pred, pos_label=pos)
    m_report = metrics.classification_report(actual, pred)
    # m_accuracy = metrics.accuracy_score(actual, pred)
    # print 'precision:{0:.3f}'.format(m_precision)
    # print 'recall:{0:0.3f}'.format(m_recall)
    return m_precision, m_recall, m_report


def train_clf(train_data, train_tags, alp=1.0):
    clf = MultinomialNB(alpha=alp)
    clf.fit(train_data, numpy.asarray(train_tags))
    return clf


def main():
    train_words, train_tags, test_words, test_tags, stop_words = input_data(threshold=4000, seed=5)
    train_data, test_data = vectorize(train_words, test_words, stop_words)

    s = SelectKBest(chi2, k=5000)
    train_data = s.fit_transform(train_data, train_tags)
    test_data = s.transform(test_data)

    clf = train_clf(train_data, train_tags, alp=0.1)
    prediction = clf.predict(test_data)
    prediction2 = clf.predict(train_data)
    pre, rec, report = evaluate(numpy.asarray(test_tags), prediction, 1)
    pre2, rec2, report2 = evaluate(numpy.asarray(train_tags), prediction2, 1)
    print report
    print pre2

    # 打印不匹配的结果
    # for index in range(len(prediction)):
    #     if test_tags[index] == 1 and prediction[index] == 2:
    #         print test_tags[index], prediction[index], test_words[index]


def main2():
    # fs_num_list = range(1, 10001, 100)
    # fs_num_list = range(1, 1602, 100)
    fs_num_list = range(1000, 14001, 1000)
    pre_list = []
    # fs_num_list = [379]
    rec_list = []
    f_list = []
    pre2_list = []
    acc_dict = {}
    for fs_num in fs_num_list:
        pre_mean = 0
        pre2_mean = 0
        rec_mean = 0
        for i in range(1, 3):
            train_words, train_tags, test_words, test_tags, stop_words = input_data(threshold=5000, seed=i+10)
            train_data, test_data = vectorize(train_words, test_words, stop_words)

            s = SelectKBest(chi2, k=fs_num)
            new_train_data = s.fit_transform(train_data, train_tags)
            new_test_data = s.transform(test_data)

            clf = train_clf(new_train_data, train_tags, alp=0.1)
            prediction = clf.predict(new_test_data)
            prediction2 = clf.predict(new_train_data)
            pre, rec, report = evaluate(numpy.asarray(test_tags), prediction, 1)
            pre2, rec2, report2 = evaluate(numpy.asarray(train_tags), prediction2, 1)
            pre_mean += pre
            pre2_mean += pre2
            rec_mean += rec
        pre_mean /= 2
        pre2_mean /= 2
        rec_mean /= 2
        f = 2*pre_mean*rec_mean/(pre_mean+rec_mean)

        pre_list.append(pre_mean)
        pre2_list.append(pre2_mean)
        rec_list.append(rec_mean)
        f_list.append(f)
        print fs_num
    acc_dict['Pre'] = pre_list
    acc_dict['Pre2'] = pre2_list
    acc_dict['Rec'] = rec_list
    acc_dict['F'] = f_list

    best_f = 0
    best_index = 0
    for index in range(len(f_list)):
        f = f_list[index]
        if f > best_f:
            best_f = f
            best_index = index

    best_fs = fs_num_list[best_index]
    print best_fs

    # for fs_method in ['Pre', 'Rec', 'F', 'Pre2']:
    for fs_method in ['Pre', 'Rec', 'F', 'Pre2']:
        plt.plot(fs_num_list, acc_dict[fs_method], label=fs_method)
        plt.title('trainset size test')
        plt.xlabel('trainset')
        plt.ylabel('accuracy')
        plt.xlim((200, 4000))
        plt.ylim((0.6, 1))
    plt.legend(loc='lower right', numpoints=1)
    plt.show()


if __name__ == '__main__':
    main()


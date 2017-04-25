# coding: utf-8
import jieba
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import re
from pandas import DataFrame
import io
import csv
import datetime


def string_preprocess(string):
    raw_string = string
    http_info = re.compile('[a-zA-z]+://[^\s]*')
    string_without_http = http_info.sub(ur'链接', raw_string)
    at_info = re.compile(ur'@[^ @，,。.]*')
    string_without_http_and_at = at_info.sub(ur'@', string_without_http)
    number_eng_info = re.compile(ur'[0-9|a-zA-Z]')
    clean_string = number_eng_info.sub('', string_without_http_and_at)
    return clean_string


def classify(traindata_filename=ur'D:\workspace\weibo\data\8000条测试数据.xlsx',
             stopwords_filename=ur"D:\workspace\weibo\data\stop_words.txt",
             newdata_filename=ur"D:\WorkSpace\Data\clean_data.txt", save_file_path=ur'data'):
    # 读取训练数据
    filename = traindata_filename
    threshold = 5000
    dataframe = pd.read_excel(filename, sheetname=2, index_col=None, header=None)
    rdata = dataframe[:threshold]
    train_words = [string_preprocess(string) for string in rdata[2]]
    train_tags = [tag for tag in rdata[1]]

    # 读取要分类的数据
    names = ['PostID', 'PublishDate', 'Content', 'Source', 'PostGeoLocation', 'PicUrls',
             'UserID', 'UserProvinceCode', 'UserCityCode', 'UserLocation', 'UserGender',
             'FollowerCount', 'FollowingCount', 'UpdateCount', 'FavouritesCount', 'Verified',
             'VerifiedReason', 'MutualFollowCount']

    filename = newdata_filename
    with io.open(filename, "r", encoding='utf-8') as f:
        # line = f.readline()
        data = [line.rstrip('\n').rstrip(' ').rstrip('\t').split("\t") for line in f]
    df = DataFrame(data, columns=names)
    new_words = [string_preprocess(string) for string in df['Content']]

    # 从文件导入停用词表
    with io.open(stopwords_filename, 'r', encoding='utf-8') as f:
        stpwrd_content = f.read()
        stop_words = stpwrd_content.splitlines()

    # 文档向量化
    # v = HashingVectorizer(tokenizer=lambda x: jieba.cut(x, cut_all=True), n_features=30000, non_negative=True,
    #                       stop_words=stop_words)
    v = TfidfVectorizer(tokenizer=lambda x: jieba.cut(x), analyzer='word', stop_words=stop_words)
    train_data = v.fit_transform(train_words)
    test_data = v.transform(new_words)
    words = v.get_feature_names()

    # 降维
    S = SelectKBest(chi2, k=5000)
    new_train_data = S.fit_transform(train_data, train_tags)
    new_test_data = S.transform(test_data)

    # 训练模型
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, np.asarray(train_tags))

    # 分类
    prediction = clf.predict(test_data)
    np.savetxt(save_file_path + ur'\prediction.txt', prediction, delimiter='\t', fmt='%d')
    # 筛选
    clean_index = []
    trash_index = []
    for index in range(len(prediction)):
        if prediction[index] == 1:
            clean_index.append(True)
            trash_index.append(False)
        else:
            clean_index.append(False)
            trash_index.append(True)
    # 保存
    clean_data = df[clean_index]
    trash_data = df[trash_index]
    clean_data.to_csv(save_file_path + ur'\clean_data.txt', header=None, encoding=u'utf-8',
                      index=None, sep='\t', mode='w', quoting=csv.QUOTE_NONE)
    trash_data.to_csv(save_file_path + ur'\trash_data.txt', header=None, encoding=u'utf-8',
                      index=None, sep='\t', mode='w', quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    starttime = datetime.datetime.now()

    classify()

    endtime = datetime.datetime.now()
    interval = endtime - starttime
    print interval

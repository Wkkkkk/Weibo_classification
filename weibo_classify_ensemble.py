# coding: utf-8
import io
import jieba
import pandas as pd
from sklearn import metrics
import numpy
import re
import time
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def string_preprocess(string):
    raw_string = string
    http_info = re.compile('[a-zA-z]+://[^\s]*')
    string_without_http = http_info.sub(ur'链接', raw_string)
    at_info = re.compile(ur'@[^ @，,。.]*')
    string_without_http_and_at = at_info.sub(ur'@', string_without_http)
    number_eng_info = re.compile(ur'[0-9|a-zA-Z|-]')
    clean_string = number_eng_info.sub('', string_without_http_and_at)
    return clean_string


def svm_classifier(train_x, train_y):
    model = SVC(kernel='rbf', C=1000, gamma=0.001, probability=True)
    model.fit(train_x, train_y)
    return model


def svm_cross_validation(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def naive_bayes_classifier(train_x, train_y):
    model = MultinomialNB(alpha=0.03)
    model.fit(train_x, train_y)
    return model


def nb_cross_validation(train_x, train_y):
    model = MultinomialNB(alpha=0.02, fit_prior=True)
    param_grid = {'alpha': [x/1000.0 for x in range(1, 1000, 37)]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = MultinomialNB(alpha=best_parameters['alpha'], fit_prior=True)
    model.fit(train_x, train_y)
    return model


def knn_classifier(train_x, train_y):
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2', C=10000, tol=0.001)
    model.fit(train_x, train_y)
    return model


def lr_cross_validation(train_x, train_y):
    model = LogisticRegression(penalty='l2')
    param_grid = {'penalty': ['l1', 'l2'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 2000, 5000, 10000, 20000, 40000],
                  'tol': [x/10000.0 for x in range(1, 11)]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = LogisticRegression(penalty=best_parameters['penalty'], tol=best_parameters['tol'], C=best_parameters['C'])
    model.fit(train_x, train_y)
    return model


def random_forest_classifier(train_x, train_y):
    model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=100)
    model.fit(train_x, train_y)
    return model


def rf_cross_validation(train_x, train_y):
    model = RandomForestClassifier(n_estimators=8)
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [100, 500, 1000, 2000, 5000, 10000],
                  'n_estimators': [10, 20, 30, 40]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = RandomForestClassifier(n_estimators=best_parameters['n_estimators'],
                                   max_depth=best_parameters['max_depth'],
                                   criterion=best_parameters['criterion']
                                   )
    model.fit(train_x, train_y)
    return model


def decision_tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1000)
    model.fit(train_x, train_y)
    return model


def dt_cross_validation(train_x, train_y):
    model = tree.DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [100, 500, 1000, 2000, 5000, 10000],
                  }
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = tree.DecisionTreeClassifier(max_depth=best_parameters['max_depth'], criterion=best_parameters['criterion'])
    model.fit(train_x, train_y)
    return model


def gradient_boosting_classifier(train_x, train_y):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':

    # region 数据读入
    threshold = 4000
    filename = ur'data\8000条测试数据.xlsx'
    df = pd.read_excel(filename, sheetname=2, index_col=None, header=None)
    rdata = df[:threshold]
    data = list(rdata[2])
    target = list(rdata[1])
    data = [string_preprocess(cur_string) for cur_string in data]

    with io.open(ur"data\stop_words.txt", 'r', encoding='utf-8') as f:
        stpwrd_content = f.read()
        stpwrdlst = stpwrd_content.splitlines()
    # endregion

    # region 建立方法字典
    # 选用的模型
    # test_classifiers = ['NB', 'LR', 'LRCV', 'SVM', 'SVMCV', 'DT', 'DTCV', 'RF', 'RFCV', 'GBDT']
    test_classifiers = ['NB', 'LR', 'SVM', 'DT', 'RF', 'GBDT']
    # 所有的模型
    classifiers = {'NB': naive_bayes_classifier,
                   'NBCV': nb_cross_validation,
                   'KNN': knn_classifier,
                   'LR':  logistic_regression_classifier,
                   'LRCV': lr_cross_validation,
                   'RF': random_forest_classifier,
                   'RFCV': rf_cross_validation,
                   'DT': decision_tree_classifier,
                   'DTCV': dt_cross_validation,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier,
                   }
    # endregion

    # region 建立模型，训练
    # v = HashingVectorizer(tokenizer=lambda x: jieba.cut(x, cut_all=True), n_features=30000, non_negative=True,
    #                       stop_words=stpwrdlst)
    v = TfidfVectorizer(tokenizer=lambda x: jieba.cut(x, cut_all=True), stop_words=stpwrdlst)
    hash_data = v.fit_transform(data)
    words = v.get_feature_names()

    # 挑选特征
    S = SelectKBest(chi2, k=5000)
    hash_data = S.fit_transform(hash_data, target)

    # 训练集和测试集
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(hash_data, target,
                                                                         test_size=0.25, random_state=1)
    y_train = numpy.asarray(y_train)

    outcome = []
    for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        start_time = time.time()

        # 训练模型
        each_model = classifiers[classifier](X_train, y_train)
        print 'training took %fs!' % (time.time() - start_time)

        # 预测
        predict = each_model.predict(X_test)
        precision = metrics.precision_score(y_test, predict, pos_label=1)
        recall = metrics.recall_score(y_test, predict, pos_label=1)
        # print metrics.classification_report(y_test, predict)
        f1 = 2 * precision * recall / (precision + recall)

        precision2 = metrics.precision_score(y_test, predict, pos_label=2)
        recall2 = metrics.recall_score(y_test, predict, pos_label=2)
        f2 = 2 * precision2 * recall2 / (precision2 + recall2)

        accuracy = metrics.accuracy_score(y_test, predict)
        print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
        print 'accuracy: %.2f%%' % (100 * accuracy)

        # 保存预测结果
        outcome.append([classifier, '1', precision, recall, f1])
        outcome.append([classifier, '2', precision2, recall2, f2])
    print outcome

    # 保存数据
    df = DataFrame(outcome)
    df.to_csv(r'outcome.txt', header=['classifier', 'type', 'precision', 'recall', 'accuracy'],
              encoding=u'utf-8', index=None, sep='\t', mode='w')
    # endregion

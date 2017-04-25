'''
@author: Garvin
'''
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from pandas import DataFrame


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def plotBestFit(dataArr1, dataArr2):
    n = shape(dataArr1)[0]
    n1 = shape(dataArr2)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        xcord1.append(dataArr1[i][0])
        ycord1.append(dataArr1[i][1])
    for i in range(n1):
        xcord2.append(dataArr2[i][0])
        ycord2.append(dataArr2[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    mata = loadDataSet(r'D:\workspace\weibo\txtoutput.txt')

    U, s, V = np.linalg.svd(mata, full_matrices=False)
    print U.shape, V.shape, s.shape
    S = np.diag(s)

    df = DataFrame(np.array(U))
    df.to_csv(r'U.txt', header=None, encoding=u'utf-8', index=None, sep='\t', mode='w')
    df = DataFrame(np.array(S))
    df.to_csv(r'S.txt', header=None, encoding=u'utf-8', index=None, sep='\t', mode='w')
    df = DataFrame(np.array(V))
    df.to_csv(r'V.txt', header=None, encoding=u'utf-8', index=None, sep='\t', mode='w')
    Mproj2 = np.dot(U[:, :2], S[:2, :2])
    x = Mproj2[:, 0].transpose().tolist()[0]
    y = Mproj2[:, 1].transpose().tolist()[0]

    with open(r'D:\workspace\weibo\data\train_tags.txt', "r") as f:
        tags = [int(line.strip()) for line in f]
    tags = array(tags)

    plt.scatter(x, y, marker='.')

    # plt.plot(x, y, 'ko')
    plt.show()

    exit()
    S = s[:2]
    # print np.allclose(mata, np.dot(U[:, :6], np.dot(S, V[:6, :])))
    U2 = U[:, :2]
    V2 = V[:2, :]
    print U2.shape, V2.shape, S.shape
    np.savetxt(r'U2.txt', U2, delimiter='\t')
    np.savetxt(r's.txt', s, delimiter='\t')
    a = []
    b = []
    with open(r'U2.txt', "r") as f:
        U = [line.rstrip('\n').rstrip(' ').rstrip('\t').split("\t") for line in f]
    with open(r'train_tags.txt', "r") as f:
        tags = [line.strip() for line in f]
    s1 = tags[0]
    for index in range(640):
        if tags[index] == s1:
            a.append(U[index])
        else:
            b.append(U[index])
    plotBestFit(a, b)

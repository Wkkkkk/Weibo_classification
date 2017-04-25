# -*- coding:utf-8 -*-
import sys, urllib, re, os
import re
import pandas as pd
from pandas import DataFrame, Series
import jieba


names = ['PostID','PublishDate','Content','Source','PostGeoLocation','PicUrls',
       'UserID','UserProvinceCode','UserCityCode','UserLocation','UserGender',
       'FollowerCount','FollowingCount','UpdateCount','FavouritesCount','Verified',
       'VerifiedReason','MutualFollowCount']
names2 = ['PostID', 'PublishDate', 'Content', 'Source', 'PostGeoLocation', 'PicUrls',
         'UserID', 'UserProvinceCode', 'UserCityCode', 'UserLocation', 'UserGender',
         'FollowerCount', 'FollowingCount', 'UpdateCount', 'FavouritesCount', 'Verified',
         'VerifiedReason', 'MutualFollowCount', 'keyword']
# 读入数据
path = ur'D:\workspace\Data\WeiboData'
file_names = os.listdir(path)
for i in range(1, 50):
    print file_names[i]
    # filename = 'D:\\workspace\\Data\\WeiboData\\' + str(i) + '\\weibo1.txt'
    filename = path + '\\' + file_names[i] + ur'\keywords_data_trash.txt'
    data1 = []
    with open(filename, "r") as f:
        # line = f.readline()
        data1 = [line.rstrip("\n").split("\t") for line in f]
    rdata1 = DataFrame(data=data1, columns=names2)

    filename = path + '\\' + file_names[i] + ur'\series_data_trash.txt'
    data2 = []
    with open(filename, "r") as f:
        # line = f.readline()
        data2 = [line.rstrip("\n").split("\t") for line in f]
    rdata2 = DataFrame(data=data2, columns=names)

    filename = path + '\\' + file_names[i] + ur'\tags_data_trash.txt'
    data3 = []
    with open(filename, "r") as f:
        # line = f.readline()
        data3 = [line.rstrip("\n").split("\t") for line in f]
    rdata3 = DataFrame(data=data3, columns=names)

    rdata12 = pd.concat([rdata1, rdata2])
    rdata123 = pd.concat([rdata12, rdata3])

    filename = path + '\\' + file_names[i] + ur'\clean_data.txt'
    data = []
    with open(filename, "r") as f:
        # line = f.readline()
        data = [line.rstrip("\n").split("\t") for line in f]
    rdata = DataFrame(data=data, columns=names)

    if i == 1:
        totaldata = rdata123
        cleandata = rdata
    else:
        totaldata = pd.concat([totaldata, rdata123])
        cleandata = pd.concat([cleandata, rdata])

print 'data loaded!'
print totaldata.shape
print cleandata.shape
# 客户端词频统计
cont = list(totaldata['Source'])
wordcut = r'\n'.join(cont)
wordcuts = wordcut.split(r'\n')

result = []
# 提取来源
for i in wordcuts:
    try:
        seg_list = re.findall(">[^\n]*<", i.decode('utf-8'))
        if seg_list:
            cur_string = seg_list[0]
            result.append(cur_string)
    except:
        print("some wrong")

# 进行计数
dic_result = {}
for i in result:
    if i in dic_result:
        dd = dic_result.get(i)
        dic_result[i] = dd + 1
    else:
        dic_result[i] = 1
dic_result = sorted(dic_result.items(), key=lambda asd: asd[1], reverse=True)
dic_data = DataFrame(dic_result, columns=['keyword', 'num1'])
dic_data.to_csv('D:\\workspace\\Data\\sources1.txt', header=None, encoding=u'utf-8', index=None, sep='\t', mode='w')

# 干净客户端
cont = list(cleandata['Source'])
wordcut = r'\n'.join(cont)
wordcuts = wordcut.split(r'\n')

result = []
# 提取来源
for i in wordcuts:
    try:
        seg_list = re.findall(">[^\n]*<", i.decode('utf-8'))
        if seg_list:
            cur_string = seg_list[0]
            result.append(cur_string)
    except:
        print("some wrong")

# 进行计数
dic_result = {}
for i in result:
    if i in dic_result:
        dd = dic_result.get(i)
        dic_result[i] = dd + 1
    else:
        dic_result[i] = 1
dic_result = sorted(dic_result.items(), key=lambda asd: asd[1], reverse=True)
dic_data2 = DataFrame(dic_result, columns=['keyword', 'num2'])
dic_data2.to_csv('D:\\workspace\\Data\\sources2.txt', header=None, encoding=u'utf-8', index=None, sep='\t', mode='w')
# 合并
dic_data3 = pd.merge(dic_data, dic_data2, how='outer', on='keyword')
dic_data3 = dic_data3.fillna(0.0)


sum1 = dic_data3['num1'].sum()
sum2 = dic_data3['num2'].sum()

dic_data3['frequency1'] = dic_data3['num1']/sum1
dic_data3['frequency2'] = dic_data3['num2']/sum2

dic_data4 = dic_data3[dic_data3['frequency1'] > dic_data3['frequency2']*10][:100]
print dic_data4.head()
dic_data4.to_csv('D:\\workspace\\Data\\laji_sources.txt',
                 encoding=u'utf-8', index=None, sep='\t', mode='w')

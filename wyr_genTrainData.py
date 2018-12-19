#!/usr/ali/bin/python
# encoding: utf-8
import sys
from pyspark import SparkConf
from pyspark import SparkContext
from spark_common.util.fs_util import FileSystemUtil
import md5

if sys.version_info < (2, 6):
    import simplejson as json
else:
    import json

# pangu_item_info = 'pangu://AY54/home/dump_stage/dump_data/xinxing.yangxx/ars/news_data/sm_db_dump/2016*'
# pangu_user_info = 'pangu://AY54/sm_user_profile/final_user_tag/$day/globa_keywords_processor/'

# 写死的path
projectsDir = '/home/zhaowei.kzw/wyr/projects/'
local_user_item = projectsDir + 'pos_samples.data'
hdfs_results = ""
pangu_item_info = 'pangu://AY54/home/dump_stage/dump_data/xinxing.yangxx/ars/news_data/sm_db_dump/201404*'
pangu_user_info = 'pangu://AY54/sm_user_profile/final_user_tag/20160416/globa_keywords_processor/1*'


def readLocalFile():
    user_itemList = []
    try:
        fr = open(local_user_item, 'r')
        user_itemList = fr.readlines()
    finally:
        if fr:
            fr.close()
    return user_itemList


def ParseUserItem(line):
    line = line.rstrip('\n')
    fields = line.split('^')
    if len(fields) != 2:
        return None
    userId = fields[0]
    itemId = fields[1]
    m1 = md5.new()
    m1.update(itemId)
    itemId = m1.hexdigest()
    return (userId, itemId)


# 解析item的信息 (itemId, itemInfo)
def ParseItemInfo(in_record):
    try:
        parts = in_record.split('\t', 1)
        # itemID = ‘news_XXXXXXXX(XXXXXXXX为item url的md5值)’
        itemId = parts[0].replace('news_', '')
        infoStr = parts[1]
        inforStrVisual = parseItemInfoVisual(infoStr)
        if len(itemId) > 0 and len(inforStrVisual) > 0:
            return (itemId, inforStrVisual)
        else:
            return None
    except:
        return None


# item 解析 title , term_weight
def parseItemInfoVisual(itemInfo):
    try:
        # print "itemInfo --- ", itemInfo
        itemJsonObj = json.loads(itemInfo)
        title = itemJsonObj['title'].decode('unicode_escape')
        term_weight = itemJsonObj['term_weight'].decode('unicode_escape')
        itemStrVisual = '^'.join([title, term_weight])
        # print "itemStrVisual ---    ", itemStrVisual
        return itemStrVisual
    except:
        return ""


# 解析用户信息 (userId, userInfo)
def ParseUserInfo(line):
    try:
        line = line.rstrip('\n')
        field = line.split('\t')
        if len(field) != 2:
            return None
        userId = field[0]
        if len(userId) == 32:  # cookie
            return None
        userInfoStr = field[1]
        userInfoVisual = paseUserInfoVisual(userInfoStr)
        if len(userId) > 0 and len(userInfoVisual) > 0:
            return (userId, userInfoVisual)
        else:
            return None
    except:
        return None


def paseUserInfoVisual(userInfo):
    try:
        # print "userInfo --- ", userInfo
        userJsonObj = json.loads(userInfo)
        attr_keywords_list = []
        for attr, attrInfo in userJsonObj.iteritems():
            # print "attr *** ", attr
            for keyword in attrInfo['keywords_list']:
                # print keyword['dim'],keyword['key'].decode('unicode_escape')
                item = "^".join([attr, keyword['dim'], keyword['key'].decode('unicode_escape')])
                attr_keywords_list.append(item)
        userVisualStr = "|".join(attr_keywords_list)
        # print "userVisual ***      ",userVisualStr
        return userVisualStr
    except:
        return ""


# 解析数据对象成为字符串
# rdd4 is (userId, (itemId, userInfo, itemInfo))
def Convert2Text(line):
    userId = line[0]
    itemId = line[1][0]
    userInfo = line[1][1]
    itemInfo = line[1][2]
    content = "\t".join([userId, itemId, userInfo, itemInfo])
    return content
'''
By default, each transformed RDD may be recomputed each time you run an action on it.
所以每当first()的时候都会重新计算一遍，太费时间。
However, you may also persist an RDD in memory using the persist (or cache) method,
in which case Spark will keep the elements around on the cluster for much faster access the next time
you query it.There is also support for persisting RDDs on disk, or replicated across multiple nodes.
'''


if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    print "\nbegin...............\n"
    hdfs_results = sys.argv[1]
    fileUtil = FileSystemUtil()

    # RddOf_userItem = (userId, itemId)
    user_items = readLocalFile()
    RddOf_userItem = sc.Parallelize(user_items).map(ParseUserItem).filter(lambda x: x != None)
    print "\n ***User Item infor************"
    print RddOf_userItem.first()

    # RddOf_itemInfo = (itemId, itemInfo)
    RddOf_itemInfo = sc.textFile(pangu_item_info).map(ParseItemInfo).filter(lambda x: x != None)
    RddOf_itemInfo.persist()
    print "\n ***itemInfo************** \n"
    print RddOf_itemInfo.first()
    # RddOf_userInfo = (userId, userInfo)
    RddOf_userInfo = sc.textFile(pangu_user_info).map(ParseUserInfo).filter(lambda x: x != None)
    RddOf_userInfo.persist()
    print "\n***userInfo****************\n"
    print RddOf_userInfo.first()

    # rdd1 is (userId, (itemId, userInfo))
    rdd1 = RddOf_userItem.join(RddOf_userInfo)
    print "\n RDD1 - (userId, (itemId, userInfo))*********\n"
    rdd1.persist()
    print rdd1.first()

    # rdd2 is (itemId, (userId, userInfo))
    rdd2 = rdd1.map(lambda x: (x[1][0], (x[0], x[1][1])))
    rdd2.persist()
    print "\n RDD2 - (itemId, (userId, userInfo))*************\n"
    print rdd2.first()

    # rdd3 is (itemId, ((userId, userInfo), itemInfo))
    rdd3 = rdd2.join(RddOf_itemInfo)
    rdd3.persist()
    print "\n RDD3 - (itemId, ((userId, userInfo), itemInfo))*********\n"
    print rdd3.first()

    # rdd4 is (userId, (itemId, userInfo, itemInfo))
    rdd4 = rdd3.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))
    rdd4.persist()
    print "\n RDD4 - (userId, (itemId, userInfo, itemInfo))*********\n"
    print rdd4.first()

    # 如果输出文件已存在，再输出时会报错，所以要首先尝试删除下
    fileUtil.Delete(hdfs_results)
    rdd4.map(Convert2Text).saveAsTextFile(hdfs_results)
    print "success ..................."

#!/usr/bin/env python
#coding=utf-8
'''
File Name: wyr_LDA.py
Author: yunrong.wyr 
mail: yunrong.wyr@alibaba-inc.com 
Created Time: Thu May 12 19:06:22 CST 2016 
'''
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

from pyspark import SparkConf
from pyspark import SparkContext
from spark_common.util.fs_util import FileSystemUtil
import spark_common.ali_ws as ali_ws
import md5

if sys.version_info < (2, 6):
    import simplejson as json
else:
    import json

# 写死的path
hdfs_results = "/user/zhaowei.kzw/smk/wyr/trainData"
pangu_item_info = 'pangu://AY54/home/dump_stage/dump_data/xinxing.yangxx/ars/news_data/sm_db_dump/2016050511'


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


# item 解析 title , content
def parseItemInfoVisual(itemInfo):
    try:
        itemJsonObj = json.loads(itemInfo)
        title = itemJsonObj['title']
        content = itemJsonObj['content']
        itemStrVisual = '-'.join([title, content])
        return itemStrVisual
    except:
        return ""

#词语切分 - 拼接字符串
def SegWord(itemID,itemInfo,ws):
    try:
        title, content = itemInfo.split('-',1)
        tokens = cal_ali_ws_str(title, content, ws)
        formatStr=""
        for word, count in tokens.iteritems():
            formatStr += '%s^%d|' % (word, count)
        return itemID + "\t" + formatStr.decode('utf-8').rstrip('|')
    except:
        return None

#词语切分 - 统计词频
def cal_ali_ws_str(title, content, ws):
    # 此处使用ali_ws.SegTokenVector()而非list()
    title_seg_res = ali_ws.SegTokenVector()
    content_seg_res = ali_ws.SegTokenVector()
    title_retCode = ws.segment(title.encode('utf-8'), ali_ws.EnumEncodingSupported.UTF8, ali_ws.SegTokenType.SEG_TOKEN_SEMANTIC_MAX, title_seg_res)
    content_retCode = ws.segment(content.encode('utf-8'), ali_ws.EnumEncodingSupported.UTF8, ali_ws.SegTokenType.SEG_TOKEN_SEMANTIC_MAX, content_seg_res)
    if not title_retCode or not content_retCode:
        return None

    tokenVec = {}
    for elem in content_seg_res:
        posTagId = elem.pos_tag_id#词性id
        word = elem.word#词语
        if isUseless(posTagId):
            continue
        #utf-8编码一个汉字3个字节
        if len(word) < 2 * 3:
            continue
        if word in tokenVec:
            tokenVec[word] += 1
        else:
            tokenVec[word] = 1

    for elem in title_seg_res:
        posTagId = elem.pos_tag_id#词性id
        word = elem.word#词语
        if isUseless(posTagId):
            continue
        if len(word) < 2 * 3:
            continue
        if word in tokenVec:
            tokenVec[word] += 3
        else:
            tokenVec[word] = 3
    return tokenVec

def isUseless(posTagId):
    if ((44 == posTagId or 45 == posTagId) or
        # 数词 数量词
        (91 <= posTagId and posTagId <= 104) or
        # 标点符号
        105 == posTagId or
        # 语气词
        (67 <= posTagId and posTagId <= 70) or
        # 代词
        74 == posTagId or
        # 助词
        19 == posTagId or
        # 连词
        (20 <= posTagId and posTagId <= 22) or
        # 副词
        55 == posTagId or
        # 介词
        5 == posTagId or
        # 数语词
        9 == posTagId or
        # 时间词
        74 <= posTagId and posTagId <= 81 or
        # 助词
        56 <= posTagId and posTagId <= 66
        # 量词
        ):
        return True
    else:
        return False

if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    print "\nbegin...............\n"
    hdfs_results = sys.argv[1]
    fileUtil = FileSystemUtil()

    # Rdd_itemInfo = (itemId, itemInfo)
    Rdd_itemInfo = sc.textFile(pangu_item_info).map(ParseItemInfo).filter(lambda x: x != None)
    #Rdd_itemInfo.persist()
    print "\n ***itemInfo*** \n"
    print Rdd_itemInfo.first()

    # 此处配置文件需使用已经打开了词性说明的AliWsPosTag.conf
    ws = ali_ws.AliTokenizer("/usr/local/libdata/AliWS/conf/AliWsPosTag.conf", "INTERNET_CHN")
    #Rdd_itemSeg = (itemId, "word1^count1|word2^count2|...")
    Rdd_itemSeg = Rdd_itemInfo.map(lambda x: SegWord(x[0],x[1],ws)).filter(lambda x : x != None)
    print "\n ***Rdd_itemSeg*** \n"
    print Rdd_itemSeg.first()

    # 如果输出文件已存在，再输出时会报错，所以要首先尝试删除下
    fileUtil.Delete(hdfs_results)
    Rdd_itemSeg.saveAsTextFile(hdfs_results)
    print "success .......^_^............"



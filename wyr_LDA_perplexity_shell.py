#!/usr/ali/bin/python
#encoding:utf-8

import sys
import math
from spark_common.util.fs_util import FileSystemUtil
from pyspark.odps import OdpsOps
from pyspark import SparkConf
from pyspark import SparkContext
from operator import add

reload(sys)
sys.setdefaultencoding('utf-8')

ctrlA = chr(1)
ctrlB = chr(2)
ctrlC = chr(3)
TAB = '\t'

def dt_parse(info):
    doc_id = info[0]
    topic_id = info[1]
    value = info[2]
    return doc_id, ctrlC.join([str(topic_id), str(value)])

def tw_parse(info):
    #topic_id 在odps上存储的是bigint形式
    topic_id = str(info[0])
    word = info[1]
    value = float(info[2])
    if value >= 1:
        return topic_id, word, value
    else:
        return None

def count_doc_words(doc):
    doc_id = doc[0]
    words_weights = doc[1]
    if words_weights == None:
        return 0
    size = len(words_weights.split(ctrlA))
    return size



def parse_topic_word(tw_dic):
    topic_word = []
    for topic, words_weis in tw_dic.iteritems():
        for word, wei in words_weis.iteritems():
            topic_word.append(TAB.join([topic, word, str(wei)]))
    return topic_word

def pre_topic_word(topic_words):
    tw_dic = {}
    for tw in topic_words:
        if tw_dic.has_key(tw[0]):
            tw_dic[tw[0]][tw[1]] = tw[2]
        else:
            tw_dic[tw[0]] = {}
            tw_dic[tw[0]][tw[1]] = tw[2]
    #topic-word 数据归一化
    for topic, words_weis in tw_dic.iteritems():
        sum_topic = 0.0
        for word, wei in words_weis.iteritems():
            sum_topic += wei
        for word, wei in words_weis.iteritems():
            tw_dic[topic][word] = wei / sum_topic
    return tw_dic

def compute_doc_perplexity(doc, tw_dic):
    doc_id = doc[0]
    doc_content = doc[1][0]
    #存在一些坏数据：doc的内容为None
    if doc_content == None:
        return (doc_id, 0.0)

    words_weights = doc_content.split(ctrlA)
    words = []
    for word_wei in words_weights:
        word, wei = word_wei.split(ctrlC)
        words.append((word, int(wei) ))

    dt_dic = {}
    d_tList = doc[1][1].split(ctrlA)
    sum_dt = 0.0
    for dt in d_tList:
        t_v = dt.split(ctrlC)
        dt_dic[t_v[0]] = float(t_v[1])
        sum_dt += float(t_v[1])
    #doc-topic数据归一化 
    for topic, wei in dt_dic.iteritems():
        dt_dic[topic] = wei / sum_dt
    #print "dts -- > ", dt_dic

    doc_perplexity = 0.0
    for word, wei in words:
        word_p = 0.0
        for topic_id, topic_value in dt_dic.items():
            #当前文档不存在该topic
            if 0.0 == topic_value:
                continue
            word_p = word_p + topic_value * (tw_dic[topic_id][word])
        #如果是停用词、感叹词、、类似的未出现在wrod vector中的词，直接过滤掉。
        if 0.0 == word_p:
            continue
        #word_p = math.log(word_p + 0.00001)
        #word_p = math.log(0.00001)
        word_p = math.log(word_p)*wei
        doc_perplexity = doc_perplexity +  word_p

    return (doc_id, doc_perplexity)

def pre_sum_doc_per(per):
    return ("perplexity", per[1])

def pre_sum_doc_per_reduce(per):
    return per[1]

def acc_perplexity(docs_perplexity, corpus_size):
    return math.exp(- (docs_perplexity / corpus_size))



if __name__ == "__main__":
    conf = SparkConf()
    conf.setAppName('wyr_LDA_perplexity_shell')
    #spark.kryoserializer.buffer.max
    conf.set('spark.kryoserializer.buffer.max', '1500m')
    sc = SparkContext(conf=conf)
    odpsOps = OdpsOps(sc, accessId, accessKey, odpsUrl, tunnelUrl)
    print "\nbegin.....odps.perplexity.. shell .......\n"

    raw_table = sys.argv[1]
    dt_table = sys.argv[2]
    tw_table = sys.argv[3]

    #raw_table = "wyr_plda_data_10day_table"
    #dt_table = "wyr_plda_out_dt_10day_10w_t100_i500_b150_table"
    #wt_table = "wyr_plda_out_wt_10day_10w_t100_i500_b150_table"
    #tw_table = "wyr_plda_out_tw_10day_10w_t100_i500_b150_table"

    #从odps中读取数据,返回有10个Partition的RDD
    dt_model_nump = 50

    #读取document-topic 分布
    dt_rdd = odpsOps.readNonPartitionTable(project, dt_table, dt_model_nump).map(dt_parse).reduceByKey(lambda x, y : ctrlA.join([x, y]))
    #print rdd.collect()
    #print "partition number is: " + str(rdd.getNumPartitions())
    #print "count is: " + str(rdd.count())

    #topic_word 分布的信息只能读入driver
    tw_model_nump = 5
    tw_rdd = odpsOps.readNonPartitionTable(project, tw_table, tw_model_nump)
    tws = tw_rdd.map(tw_parse).filter(lambda x : x != None).collect()
    #预处理topic_word分布
    tw_dic = pre_topic_word(tws)
    #print "tw_dic -- > ", len(tw_dic)

    #读取文章的内容
    raw_nump = 80
    raw_rdd = odpsOps.readNonPartitionTable(project, raw_table, raw_nump)
    #统计词数
    corpus_size = raw_rdd.map(count_doc_words).reduce(add)
    #print "\n*****corpus_size*****\n", corpus_size

    #info_rdd : (doc_id, (doc_words, doc_topic))
    info_rdd = raw_rdd.join(dt_rdd)
    #print "\n*****info_rdd*****\n ", info_rdd.map(lambda x : TAB.join([x[0], x[1][0], x[1][1]])).first()

    #计算每篇文章的交叉熵
    doc_perplexity_rdd = info_rdd.map(lambda x : compute_doc_perplexity(x, tw_dic))
    #数据cache到内存
    doc_perplexity_rdd.cache()
    print "\n*****document perplexity*****\n", doc_perplexity_rdd.take(5)

    #计算训练语料的perplexity
    #docs_perplexity = doc_perplexity_rdd.map(pre_sum_doc_per).reduceByKey(add).collect()[0][1]
    docs_perplexity = doc_perplexity_rdd.map(pre_sum_doc_per_reduce).reduce(add)
    print "\n*****model perplexity*****\n", acc_perplexity( docs_perplexity,corpus_size)

    print "success .......^_^............"


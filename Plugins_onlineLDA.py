#!/usr/ali/bin/python
#coding:utf-8

import sys 
import numpy as np
import random
import codecs
from collections import OrderedDict
import cPickle
import time
from scipy.special import gammaln, psi 

reload(sys)
sys.setdefaultencoding('utf-8')

#ctrlA = chr(1)
#ctrlB = chr(2)
#ctrlC = chr(3)
#TAB = '\t'

def now():
    return time.strftime("%H:%M:%S")

def dirichlet_expectation(alpha):
    """ 
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.
    
    """
    if (len(alpha.shape) == 1): 
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype)  # keep the same precision as input

class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


class LDAModel(object):

    def __init__(self, K=10, iter=20):
        #
        #--模型参数
        #聚类个数K（话题个数），
        #迭代次数iter_times,
        #超参数α（alpha） β(beta)
        #alpha变小，是尽可能让同一个文档只有一个主题. α = K / 50
        #beta变小，是让一个词尽可能属于同一个主题。β = [0.01 - 0.1]
        #
        self.__K = K
        self.__alpha = 50.0 / self.__K
        self.__iter_times = iter
        self.gamma_threshold = 0.0001
        print "Model-Parameter: K=%d iter_times=%d"%(self.__K,self.__iter_times)
        #
        # -- 公共变量存储
        # p: ( Wi-T )               K维                  概率向量,double类型,存储采样的临时变量 - 每个词 采用K个topic的概率分布，求最大
        # W_T:( W-T 表) <-> N[k,t]  总词数(词向量)* K    每个词word在每个主题topic上的分布
        # Nk:( K * Nk )             K维度                每个topic的词的总数
        # N_mk: ( M * K * N[m,k])   M*K维                每个doc中各个topic的词的总数
        # Nm:  (M * Nm)             M维                  每各doc中词的总数
        # Z:(M * Nm)                M*Nm维度             每个词的当前主题 ,每个词分派一个类, 维度：M*docs[i].length
        #
        self.__p = np.zeros(self.__K)  # array([ 0., 0., 0.])
        self.__N_k = np.zeros(self.__K, dtype="int")

        #
        # -- 输入和输出数据 
        #doc_topics             预测文档的topic分布
        #word2id                word-id的映射词典
        #topic_words_dic        LDA模型的topic-word分布 
        #word_topics_dic        由topic-word分布转换成的word-topic分布
        #
        self.__doc_topics = np.zeros(self.__K)
        self.__word2id = OrderedDict()
        self.__word_topics_dic = {}

    def initialize(self, word_vector_file, topic_words_file):
        self.__word2id = self._get_wordVector(word_vector_file)
        self.__word_topics_dic = self._get_wordTopics(topic_words_file)

     #解析wordVector,并且映射为id
    def _get_wordVector(self, word_vector_file):
        # 读取wordVector
        word2id = OrderedDict()
        with codecs.open(word_vector_file,'r') as fr_word_vector:
            word_id = 0
            for word in fr_word_vector:
                word = word.rstrip('\n')
                word2id[word] = word_id
                word_id += 1
        print now(), "读取wordVector size=", len(word2id)
        return word2id

    #解析topic-word分布
    def _get_wordTopics(self, topic_words_file):
        # 读取Topic_Words 分布
        raw_tw_dic = {}
        with codecs.open(topic_words_file, 'r') as fr_topic_word:
            topic_word_list = fr_topic_word.readlines()
            size = len(topic_word_list)
            print now(), "topic_words size=", size
            ratio = size / 10
            i = 0
            for topic_word_value in topic_word_list:
                topic_str, word_str, wei_str = topic_word_value.split(',')
                topic, word_id, wei = self._type_cast(topic_str, word_str, wei_str)
                if raw_tw_dic.has_key(topic):
                    raw_tw_dic[topic][word_id] = wei
                else:
                    raw_tw_dic[topic] = {}
                    raw_tw_dic[topic][word_id] = wei
                i += 1
                if (i % ratio == 0):
                    print now(), "have reading T-W Distribution: %d0%%" % (i / ratio)
        nor_tw_dic = self._normalize_topic_word(raw_tw_dic)
        print now(), "归一化Topic_Words分布 topic size=", len(nor_tw_dic)
        wt_dic = self._tw_cast2_wt(nor_tw_dic)
        print now(), "转换为Word-Topic分布 word size=",len(wt_dic)
        return wt_dic

    #类型转化
    def _type_cast(self, topic_str, word_str, value_str):
        return int(topic_str), self.__word2id[word_str], float(value_str)

    #对topic-word归一化处理
    # {0:{'a_id':0.1,'b_id':0.2},1:{},2:{}}
    def _normalize_topic_word(self,raw_tw_dic):
        for topic, words_weis in raw_tw_dic.iteritems():
            sum_topic = 0.0
            for word, wei in words_weis.iteritems():
                sum_topic += wei
            for word, wei in words_weis.iteritems():
                raw_tw_dic[topic][word] = wei / sum_topic
        return raw_tw_dic

    #把topic-word分布 转换为 word-topic分布
    # tw   {0:{a_id:0.1,b_id:0.2},1:{},2:{}}
    # wt   {a_id:<0.1, 0.01, 0.3...>, b_id:<>, ...}
    def _tw_cast2_wt(self, nor_tw_dic):
        wt_dic = {}
        for topic, words_weis in nor_tw_dic.iteritems():
            for word_id, wei in words_weis.iteritems():
                if wt_dic.has_key(word_id):
                    wt_dic[word_id][topic] = wei
                else:
                    wt_dic[word_id] = np.zeros(self.__K)#array()
                    wt_dic[word_id][topic] = wei
        return wt_dicset paste

    #解析document，word -> id
    def _parse_doc(self, words):
        # 生成一个文档对象
        doc = Document()
        for w in words:
            # 使用词向量过滤
            if self.__word2id.has_key(w):
                doc.words.append(self.__word2id[w])
        # 文档长度定义为过滤过后的词的个数
        doc.length = len(doc.words)
        return doc

    #初始化预测文本的topic
    def _initializeTopic(self, document):
        print "\n",now(), "初始化topic分布, 文章词数:",document.length
        #清空N_k
        self.__N_k = np.zeros(self.__K, dtype="int")
        Z = np.zeros(document.length, dtype="int")
        # 随机先分配topic
        for word_index in xrange(document.length):  # Nm
            topic = random.randint(0, self.__K - 1)
            Z[word_index] = topic  # (M * Nm)
            self.__N_k[topic] += 1  # (M * K * N[m,k])
        return Z

    def get_word_id(self, words):
        doc = {}
        for w in words:
            try:
                wordID = self.__word2id[w]
            except:
                pass
            if wordID != None:
                try:
                    doc[wordID] += 1
                except:
                    doc[wordID] = 1
        return doc

    def est_new(self, words):
        print now(), "开始预测^_^"
        gammad = np.random.gamma(100., 1. / 100., (self.__K))
        Elogthetad = dirichlet_expectation(gammad)
        expElogthetad = np.exp(Elogthetad)
        converged = 0
        doc = self.get_word_id(words)
        ids = [id for id, _ in doc.items()]
        cts = np.array([cnt for _, cnt in doc.items()])
        wordTopic = []
        check = 0
        for id in ids:
            if id == 140:
                check = len(wordTopic)
            wordTopic.append(self.__word_topics_dic[id])
        expElogbetad = (np.vstack(wordTopic).T)
        phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
        for _ in xrange(self.__iter_times):
            lastgamma = gammad
            gammad = self.__alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
            Elogthetad = dirichlet_expectation(gammad)
            expElogthetad = np.exp(Elogthetad)
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            meanchange = np.mean(abs(gammad - lastgamma))
            if (meanchange < self.gamma_threshold):
                converged += 1
               # break
        self.__doc_topics = gammad
        print now(), "预测完成^_^"
        return gammad

    #预测文本的topic分布
    def est(self, words):
        print now(), "开始预测^_^"
        document = self._parse_doc(words)
        Z = self._initializeTopic(document)
        for i in xrange(self.__iter_times):
            #print now(), "The %d iteration" % i
            for word_j in xrange(document.length):
                topic = self.sampling(document, Z, word_j)
                Z[word_j] = topic

        # 计算文章-主题分布
        self._doc_topics(document.length)
        print now(), "预测完成^_^"

    #采样word_j的topic
    def sampling(self, document, Z, word_j):
        topic = Z[word_j]
        word = document.words[word_j]
        self.__N_k[topic] -= 1
        N = document.length - 1

        Kalpha = self.__K * self.__alpha
        # p: K维向量
        word_KTopics = self.__word_topics_dic[word]
        self.__p = ((self.__N_k + self.__alpha) / (N + Kalpha)) * word_KTopics

        for k in xrange(1, self.__K):
            self.__p[k] += self.__p[k - 1]
        # 模拟Mul(p)采样topic
        u = random.uniform(0, self.__p[self.__K - 1])
        for topic in xrange(self.__K):
            if self.__p[topic] > u:
                break

        self.__N_k[topic] += 1
        return topic

    #计算后验分布
    def _doc_topics(self, N):
        # 后验分布
        self.__doc_topics = (self.__N_k + self.__alpha) / (N + self.__K * self.__alpha)

    #获取doc-topic分布按概率大小排序后的List
    def get_sortedTopicList(self):
        sortedTopicList = []
        topic_dict = {}
        topic_id = 0
        for topic_value in self.__doc_topics:
            topic_dict[topic_id] = topic_value
            topic_id += 1
            sortedTopicList = sorted(topic_dict.iteritems(), key=lambda x: x[1], reverse=True)
        return sortedTopicList

    #获取doc-topic分布按概率大小排序后的OrderedDict
    def get_sortedTopicDict(self):
        sortedTopicDict = OrderedDict()
        sortedTopicList = self.get_sortedTopicList()
        for topic, value in sortedTopicList:
            sortedTopicDict[topic] = value
        return sortedTopicDict

    def get_message(self):
        print "LDA Model Parameter:"
        print "K =", self.__K
        print "alpha =", self.__alpha
        print "iter_times =", self.__iter_times

    # 缓存 word  - id 映射
    def cachewordidmap(self):
        with codecs.open("word_id_map_file", 'w') as fw:
            for word, wordId in self.__word2id.iteritems():
                line = '%s\t%s\n' % (word, str(wordId))
                fw.write(line)
        print now(), "word-id Map已保存到%s" % "word_id_map_file"

    def cacheparamfile(self):
        # 保存参数设置
        with codecs.open("paramfile", 'w', ) as fw:
            fw.write('K=' + str(self.__K) + '\n')
            fw.write('alpha=' + str(self.__alpha) + '\n')
            fw.write('iter_times=' + str(self.__iter_times) + '\n')
        print now(), "参数设置已保存到%s" % "paramfile"

def preprocessing():
    # 读取预测文本
    words_list = []
    with codecs.open("inputData/trainfile", 'r') as fr_train:
        print now(), "读取预测文本"
        docs = fr_train.readlines()
    for doc in docs:
        words_list.append(doc.split()[1:])
    return words_list

def run():
    docs = preprocessing()
    #lda = LDAModel(K=10, iter=10)
    lda = LDAModel(K=600, iter=100)
    lda.get_message()
    #lda.initialize(word_vector_file="inputData/word_vector", topic_words_file="inputData/topic_words_t10")
    lda.initialize(word_vector_file="inputData/word_vector", topic_words_file="inputData/topic_words_t600")
    for doc in docs:
        lda.est_new(doc)
        print lda.get_sortedTopicList()[:5]
        #print lda.get_sortedTopicDict()

if __name__ == '__main__':
    run()




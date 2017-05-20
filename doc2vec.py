#!/usr/bin/env python

'''
这个文件提供将文本转化为向量的能力，使用的是Jieba进行分词

sudo pip3 install jieba


使用gensim中的doc2vec模型进行文本到向量的转换

sudo pip3 install gensim

'''
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import os


def singleton_decorator(func):
    '''
    单例模式装饰器保证每次运行程序的查询不会载入模型多次
    '''
    model = {}
    def _(name):
        if name not in model:
            model[name] = func(name)
        return model[name]
    return _


def get_stop_words(file_path="./stopwords.txt"):
    with open(file_path, "r") as f:
        lines =[line.strip() for line in f.readlines()]
    return lines
stopwords = get_stop_words()
stopwords = [u""+word for word in stopwords]


class LabeledLineSentence(object, ):

    '''
    使用jieba分词从原始文本中提取出词序列，提供给d2v模型训练
    '''

    def __init__(self, txt_dir_name):
        self.txt_dir_name = txt_dir_name
        self.files_names = os.listdir(self.txt_dir_name)
        #self.files_names = os.listdir(self.txt_dir_name)[:20]

    def __iter__(self):
        for index, name in enumerate(self.files_names):
            try:
                words = self.get_words(name)
                yield LabeledSentence(words, tags=[name])
            except Exception as e:
                print("some error while reading raw file")
                print(e)

    def get_words(self, name):
        with open(self.txt_dir_name+"/{0}".format(name), "r") as f:
            line = f.readline()
        line = line.strip()
        #print(line)
        words = jieba.cut(line)
        return [word for word in words if word not in stopwords]


def get_d2v_model(txt_dir_name, vec_dimension=256, iter_=10, min_count=5,
                  model_name="d2vModel"):
    '''
    参数列表：
    vec_dimension: 文档向量的维度
    iter_: 算法迭代的次数
    min_count: 出现单词的最小次数
    model_name: 模型的名称，会在当前目录下保存这个模型

    返回：
    d2vModel
    '''

    print("training model")
    model = Doc2Vec(LabeledLineSentence(txt_dir_name), size=vec_dimension, iter=iter_, min_count=min_count)
    model.delete_temporary_training_data() # 删除训练数据，减少内存开销
    model.save(model_name)
    return model




@singleton_decorator
def load_d2v_model(model_name):
    model = Doc2Vec.load(model_name)
    return model


def query(d2v_model_name, txt_file_name):
    with open(txt_file_name, "r") as f:
        line = f.readline().strip()
    words = [word for word in jieba.cut(line) if word not in stopwords]
    model = load_d2v_model(d2v_model_name)
    return model.infer_vector(words)


if __name__ == "__main__":
    txt_dir_name= "/media/rw/DATA/sogou/formalCompetition4/News_info_train"
    '''
    for words in LabeledLineSentence(txt_dir_name):
        print(words)
    '''
    get_d2v_model(txt_dir_name)
    load_d2v_model("d2vModel")
    print(query("d2vModel", "/media/rw/DATA/sogou/formalCompetition4/News_info_train/2016999966.txt"))

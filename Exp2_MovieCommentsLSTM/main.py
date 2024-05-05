# 导入paddle及相关包
import paddle
import paddle.nn.functional as F
import re
import random
import tarfile
import requests
import numpy as np

paddle.seed(0)
random.seed(0)
np.random.seed(0)
print(paddle.__version__)


def download():
    # 通过python的requests类，下载存储在
    # https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz的文件
    corpus_url = "https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz"
    web_request = requests.get(corpus_url)
    corpus = web_request.content

    # 将下载的文件写在当前目录的aclImdb_v1.tar.gz文件内
    with open("./aclImdb_v1.tar.gz", "wb") as f:
        f.write(corpus)
    f.close()


download()


# 读取数据
def load_imdb(is_training):

    # 将读取的数据放到列表data_set里
    data_set = []

    # data_set中每个元素都是一个二元组：(句子，label)，其中label=0表示消极情感，label=1表示积极情感
    for label in ["pos", "neg"]:
        with tarfile.open("./aclImdb_v1.tar.gz") as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)
            tf = tarf.next()
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label))
                tf = tarf.next()

    return data_set

train_corpus = load_imdb(True)
test_corpus = load_imdb(False)

# 打印第一条数据，查看数据格式：(句子，label)
print(train_corpus[0])
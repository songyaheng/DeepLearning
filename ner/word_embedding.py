from gensim.models import word2vec
import sys
if __name__ == '__main__':
    sentences = word2vec.Text8Corpus("{}/chars_text.txt".format(sys.argv[1]))  # 加载语料
    model = word2vec.Word2Vec(sentences, sg=1, size=100, alpha=0.025, min_count=0, window=5, max_vocab_size=None, sample=0.001, iter=200)  # 训练skip-gram模型; 默认window=5
    # #获取“学习”的词向量
    # print("学习：" + model["学习"])
    # # 计算两个词的相似度/相关程度
    # y1 = model.similarity("不错", "好")
    # # 计算某个词的相关词列表
    # y2 = model.most_similar("书", topn=20)  # 20个最相关的
    # # 寻找对应关系
    # print("书-不错，质量-")
    # y3 = model.most_similar(['质量', '不错'], ['书'], topn=3)
    # # 寻找不合群的词
    # y4 = model.doesnt_match("书 书籍 教材 很".split())
    # 保存模型，以便重用
    model.save("{}/embeding.model".format(sys.argv[1]))
    # 对应的加载方式
    # model = word2vec.Word2Vec.load("db.model")
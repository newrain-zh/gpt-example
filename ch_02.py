import jieba
import numpy as np
import matplotlib.pyplot as plt

corpus = ["我特别特别喜欢看电影",
          "这部电影真的是很好看的电影",
          "今天天气真好是难得的好天气",
          "我今天去看了一部电影",
          "电影院的电影都很好看"]

#  分词，将结果转换成列表
corpus_tokenized = [list(jieba.cut(sentence)) for sentence in corpus]

# 创建词汇表
word_dict = {}  # 初始化词汇表
for sentence in corpus_tokenized:
    for word in sentence:
        # 如果词汇表中没有该词，则将其添加到词汇表中
        if word not in word_dict:
            word_dict[word] = len(word_dict)  # 分配当前词汇表索引

print("词汇表：", word_dict)

# 生成词袋表示
bow_vectors = []
# 遍历分词后的语料库
for sentence in corpus_tokenized:
    sentence_vectors = [0] * len(word_dict)
    for word in sentence:
        sentence_vectors[word_dict[word]] += 1
    bow_vectors.append(sentence_vectors)
print("词袋表示：", bow_vectors)

def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(v1, v2)  # 计算向量的点积
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)  # 返回余弦的相似度


# 初始化一个全 0 矩阵，用于存储余弦的相似度
similarity_matrix = np.zeros((len(corpus), len(corpus)))
for i in range(len(corpus)):
    for j in range(i, len(corpus)):
        similarity_matrix[i][j] = cosine_similarity(bow_vectors[i], bow_vectors[j])

plt.rcParams["font.family"] = ['Source Han Sans SC']
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
# cax = ax.matshow(similarity_matrix,cmap=plt.cm.Blues)
cax = ax.matshow(similarity_matrix, cmap='Blues')
fig.colorbar(cax)
ax.set_xticks(range(len(corpus)))
ax.set_yticks(range(len(corpus)))
ax.set_xticklabels(corpus,rotation=45,ha='left')
ax.set_yticklabels(corpus)
plt.show()

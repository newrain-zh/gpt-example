import jieba
import numpy as np
import matplotlib.pyplot as plt

#词袋模型示例


# corpus = ["我特别特别喜欢看电影",
#           "这部电影真的是很好看的电影",
#           "今天天气真好是难得的好天气",
#           "我今天去看了一部电影",
#           "电影院的电影都很好看"]
corpus = ["猫追老鼠",
          "老鼠追猫",
          ]

#  分词，将结果转换成列表
corpus_tokenized = [list(jieba.cut(sentence)) for sentence in corpus]
print(f"分词结果{corpus_tokenized}")

# 创建词汇表
word_dict = {}  # 初始化词汇表
for sentence in corpus_tokenized:
    for word in sentence:
        # 如果词汇表中没有该词，则将其添加到词汇表中
        if word not in word_dict:
            word_dict[word] = len(word_dict)  # 分配当前词汇表索引

print(f"词汇表：{word_dict}")

# 生成词袋表示
bow_vectors = []
# 遍历分词后的语料库
for sentence in corpus_tokenized:
    sentence_vectors = [0] * len(word_dict)
    for word in sentence:
        sentence_vectors[word_dict[word]] += 1
    bow_vectors.append(sentence_vectors)
# [1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#  向量中的每个元素对应词在文本中出现的次数。
print(f"词袋表示：{bow_vectors}")


# 定义余弦相似度函数
def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(v1, v2)  # 计算向量的点积
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)  # 返回余弦的相似度


# 初始化一个全 0 矩阵，用于存储余弦的相似度
similarity_matrix = np.zeros((len(corpus), len(corpus)))
# 计算每两个句子之间的余弦相似度
for i in range(len(corpus)):
    for j in range(i, len(corpus)):
        print(f"bow_vectors[i]:{bow_vectors[i]} bow_vectors[j]:{bow_vectors[j]}")
        similarity_matrix[i][j] = cosine_similarity(bow_vectors[i], bow_vectors[j])

plt.rcParams["font.family"] = ['Source Han Sans SC']
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()  # 创建一个绘图对象
# cax = ax.matshow(similarity_matrix,cmap=plt.cm.Blues)
cax = ax.matshow(similarity_matrix, cmap='Blues')  # 使用 matshow函数绘制余弦相似度矩阵
fig.colorbar(cax)  # 条形图颜色映射
ax.set_xticks(range(len(corpus)))  # 设置 x 轴刻度
ax.set_yticks(range(len(corpus)))  # 设置 y 轴刻度
ax.set_xticklabels(corpus, rotation=45, ha='left')
ax.set_yticklabels(corpus)
plt.show()

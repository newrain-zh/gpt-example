# 生成 CBOW 训练数据
import torch

sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
             "Xiaobing is Student", "Xiaoxue is Student",
             # "Apple is a mobile phone brand",
             # "Apples are a fruit"
             ]
# 将所有句子连接在一起，然后用空格分隔成多个单词
words = '  '.join(sentences).split()
# 构建词汇表，去除重复的词
word_list = list(set(words))
# 创建一个字典，将每个词映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建一个字典，将每个索引映射到对应的词
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)  # 计算词汇表的大小
print(f"词汇表：word_list= {word_list}")  # 输出词汇表
print(f"词汇到索引的字典：word_to_idx= {word_to_idx}")  # 输出词汇到索引的字典
print(f"索引到词汇的字典：idx_to_word={idx_to_word}")  # 输出索引到词汇的字典
print("词汇表大小：", voc_size)  # 输出词汇表大小


def create_cbow_dataset(sentences, window_size=2):
    data = []  # 初始化数据
    for sentence in sentences:
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取上下文词汇 将当前单词前后各 window_size 个单词作为周围词
            context_words = sentence[max(idx - window_size, 0):idx] + sentence[
                idx + 1:min(idx + window_size + 1, len(sentence))]
            # 将当前单词与上下文词汇作为一组训练数据
            data.append((word, context_words))
    return data  # 返回 CBOW 训练数据


cbow_data = create_cbow_dataset(sentences)
print(f"CBOW 训练数据示例: {cbow_data[:5]}")


# oneHot编码
# 将单词转换为 one-hot向量表示
def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx))  # 创建一个长度与词汇表相同的全0张量
    tensor[word_to_idx[word]] = 1  # 将对应词索引位置上的值设为1
    return tensor  # 返回生成的One-Hot编码后的向量


# 定义CBOW模型
import torch.nn as nn


class CBOW(nn.Module):

    def __init__(self, voc_size, embedding_size):
        super(CBOW, self).__init__()
        # 从词汇表大小到嵌入层大小（维度）的线性层（权重矩阵）
        # 将 one-hot 编码的单词映射到低维嵌入空间
        # 把 one-Hot 编码的向量从词汇表大小映射到嵌入层大小，以形成并学习词的向量表示
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        # 从嵌入层大小（维度）到词汇表大小的线性层（权重矩阵）
        # 将嵌入向量映射回高维词汇表空间 用于预测上下词
        # 把词的向量表示从嵌入层大小映射回词汇表大小，以预测目标词
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        # 生成嵌入
        embeddings = self.input_to_hidden(X)
        # 计算隐藏层 求嵌入的均值[embedding_size]
        hidden_layer = torch.mean(embeddings, dim=0)
        # 生成输出层
        out_layer = self.hidden_to_output(hidden_layer.unsqueeze(0))
        return out_layer


embedding_size = 2  # 设定嵌入层的大小 这里选择 2 是为了方便展示
cbow_model = CBOW(voc_size, embedding_size)
print("CBOW 模型参数:", cbow_model)
criterion = nn.CrossEntropyLoss()
import torch.optim as optim  # 导入随机梯度下降优化器

learning_rate = 0.001  # 设置学习速率
optimizer = optim.SGD(cbow_model.parameters(), lr=learning_rate)
epochs = 1000
loss_values = []  # 用于存储每轮的平均损失值
for epoch in range(epochs):
    loss_sum = 0  # 初始化损失值
    for target, context_words in cbow_data:
        X = torch.stack([one_hot_encoding(word, word_to_idx) for word in context_words]).float()
        y_true = torch.tensor([word_to_idx[target]], dtype=torch.long)
        # 模型的输出 ，是预测值
        y_pred = cbow_model(X)  # 计算预测值
        loss = criterion(y_pred, y_true)  # 计算损失
        loss_sum += loss.item()  # 累积损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    if (epoch + 1) % 100 == 0:  # 输出每100轮的损失，并记录损失
        print(f"Epoch: {epoch + 1}, Loss: {loss_sum / len(cbow_data)}")
        loss_values.append(loss_sum / len(cbow_data))

# 绘制训练损失曲线
import matplotlib.pyplot as plt  # 导入matplotlib

# 绘制二维词向量图
plt.rcParams["font.family"] = ['Source Han Sans SC']  # 用来设定字体样式
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']  # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(range(1, epochs // 100 + 1), loss_values)  # 绘图
plt.title('训练损失曲线')  # 图题
plt.xlabel('轮次')  # X轴Label
plt.ylabel('损失')  # Y轴Label
plt.show()  # 显示图
print("\n")
print("COBW词嵌入：")
for word, idx in word_to_idx.items():  # 输出每个词的嵌入向量
    print(f"{word}: {cbow_model.input_to_hidden.weight[:, idx].detach().numpy()}")

fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    vec = cbow_model.input_to_hidden.weight[:, idx].detach().numpy()
    ax.scatter(vec[0], vec[1])  # 在图中绘制嵌入向量的点
    ax.annotate(word, (vec[0], vec[1]), fontsize=10)
plt.title('二维词嵌入')
plt.xlabel('向量维度1')
plt.xlabel('向量维度2')
plt.show()

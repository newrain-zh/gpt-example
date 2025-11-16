from collections import defaultdict, Counter

# 基于顺序和概率统计生成文本
corpus = ["我喜欢吃苹果", "我喜欢吃香蕉", "她喜欢吃葡萄", "他不喜欢吃香蕉", "他喜欢吃苹果", "她喜欢吃草莓"]


# corpus = ["我喜欢吃苹果"]


def tokenize(text):
    return [char for char in text]


# 统计预料库的词频
def count_ngrams(corpus, n):
    ngrams_count = defaultdict(Counter)
    for text in corpus:
        tokens = tokenize(text)  # 分词
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            prefix = ngram[:-1]
            token = ngram[-1]
            ngrams_count[prefix][token] += 1
    return ngrams_count


bigram_counts = count_ngrams(corpus, 2)
print(type(bigram_counts))
print("Bigram词频")
for prefix, counts in bigram_counts.items():
    prefix_str = "".join(prefix) if prefix else ":"
    print(f"{prefix_str} {dict(counts)}")


# 统计Bigram词频概率
def ngram_probabilities(ngram_counts):
    ngram_probs = defaultdict(Counter)
    for prefix, tokens_count in ngram_counts.items():  # 遍历 N-Grams 前缀
        total_count = sum(tokens_count.values())
        for token, count in tokens_count.items():
            ngram_probs[prefix][token] = count / total_count
    return ngram_probs


bigram_probs = ngram_probabilities(bigram_counts)
print("\nbigram出现的概率")
for prefix, probs in bigram_probs.items():
    print("{}:{}".format("".join(prefix), dict(probs)))


# 根据 Bigram出现概率随机生成下一个词
# 接受一个前缀，返回下一个词
def generate_next_token(prefix, ngram_probs):
    if not prefix in ngram_probs:
        return None
    next_token_probs = ngram_probs[prefix]
    next_token = max(next_token_probs, key=next_token_probs.get)  # 选择概率大的做为下一个词为下一个词
    return next_token


# 定义连续生成的文本
def generate_text(prefix, ngram_probs, n, length=6):
    tokens = list(prefix)
    for _ in range(length - len(prefix)):  # 根据指定长度生成文本
        # tokens[-(n - 1):] 获取前缀的前 n-1 个词
        next_token = generate_next_token(tuple(tokens[-(n - 1):]), ngram_probs)
        if not next_token:
            break
        tokens.append(next_token)
    return "".join(tokens)


generated_text = generate_text("我喜欢", bigram_probs, 2)
print("\n生成的文本： ", generated_text)

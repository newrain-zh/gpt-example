import gensim
model = gensim.models.Word2Vec([["特别", "好"], ["你好"],["非常"]], vector_size=3, min_count=1)

# 检查语义相似度
print(model.wv.similarity("特别", "你好"))  # 可能输出接近0的值
print(model.wv.similarity("特别", "非常"))  # 可能输出较高值


from sklearn.feature_extraction.text import CountVectorizer

corpus = ["特别", "你好"]
vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())  # 输出：['特别', '你好']
print((X[0] * X[1].T).toarray()[0][0])   # 点积=0（无相似性）
print(X[0].dot(X[1].T))
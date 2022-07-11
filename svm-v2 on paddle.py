
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def deal(x):
    seq = x.split()
    return seq

train_file = "./data.csv"
df = pd.read_csv(train_file)
x_train, x_test, y_train, y_test = train_test_split(df['sequence'], df['label'], test_size=0.1)
print(y_train)
print(x_train)

x_train = x_train.apply(lambda x: deal(x))
x_test = x_test.apply(lambda x: deal(x))


word2vec = Word2Vec(x_train, vector_size=50, window=3, min_count=1, sg=1, hs=1, epochs=10, workers=25)
word2vec.save('word2vec.model')


def total_vector(words):
    vec = np.zeros(50).reshape((1, 50))
    for word in words:
        try:
            vec += word2vec.wv[word].reshape((1, 50))
        except KeyError:
            continue
    return vec


train_vec = np.concatenate([total_vector(words) for words in x_train])


model = SVC(kernel='rbf', verbose=True)

model.fit(train_vec, y_train)

print("train accurancy:",model.score(train_vec, y_train))  # important 准确值
train_pre = model.predict(train_vec)  # 预测值（结果内容是识别的具体值）
print(classification_report(train_pre, y_train))  # 输出分类报告（大概就是准确率、召回率）

# 保存模型为pkl文件
joblib.dump(model, 'weather_svm.pkl')

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics



def deal(x):
    seq = x.split()
    return ' '.join(seq)

def data_processing():
    train_file = "./data.csv"
    df = pd.read_csv(train_file)
    x_train, x_test, y_train, y_test = train_test_split(df['sequence'], df['label'], test_size=0.1)
    x_train = x_train.apply(lambda x: deal(x))
    x_test = x_test.apply(lambda x: deal(x))

    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    x_train_weight = x_train.toarray()
    x_test_weight = x_test.toarray()

    return x_train_weight, x_test_weight, y_train, y_test


def train_model():
    x_train_weight, x_test_weight, y_train, y_test = data_processing()
    start = time.time()
    print("开始时间: ", start)
    model = RandomForestClassifier(max_depth=5,n_estimators=5,random_state=2)
    model.fit(x_train_weight,y_train)
    end = time.time()
    print("结束时间: ", end)
    print("花费时间: ", (end - start))
    y_predict = model.predict(x_test_weight)

    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    print('准确率：', metrics.accuracy_score(y_test, y_predict))
    print("混淆矩阵是: ", confusion_mat)
    print('分类报告:', metrics.classification_report(y_test, y_predict))

if __name__ == '__main__':
    train_model()
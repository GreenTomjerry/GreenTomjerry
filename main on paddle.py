import time
#import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import xgboost as xgb

'''
def get_stop_words():
    filename = "./data.csv"
    stop_word_list = []
    with open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            stop_word_list.append(line.strip())
    return stop_word_list

def processing_sentence(x, stop_words):
    cut_word = jieba.cut(str(x).strip())
    words = [word for word in cut_word if word not in stop_words and word != ' ']
    return ' '.join(words)
'''

def deal(x):
    seq = x.split()
    return ' '.join(seq)

def data_processing():
    train_file = "./data.csv"
    df = pd.read_csv(train_file)
    x_train, x_test, y_train, y_test = train_test_split(df['sequence'], df['label'], test_size=0.1)
    #stop_words = get_stop_words()
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
    print("start time is: ", start)
    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100,
                              silent=False, objective='binary:logistic')
    model.fit(x_train_weight, y_train)
    end = time.time()
    print("end time is: ", end)
    print("cost time is: ", (end - start))
    y_predict = model.predict(x_test_weight)

    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    print('准确率：', metrics.accuracy_score(y_test, y_predict))
    print("confusion_matrix is: ", confusion_mat)
    print('分类报告:', metrics.classification_report(y_test, y_predict))

if __name__ == '__main__':
    train_model()
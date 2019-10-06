import os
from utils import read_train_txt, read_test_txt, write_prediction, get_user_dict, get_user_dict_test
from preprocess import process_one_tweet
import numpy as np
import pickle
import random
import math

# for text processing
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize.casual import TweetTokenizer
from collections import Counter

# for machine learning
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

DATA_DIR = "data/"
TRAIN_TXT = "train_tweets.txt"
TEST_TXT = "test_tweets.txt"
OUT_CSV = "test_prediction.csv"

def main():

    # -------------- Stage 1: get user dict --------------

    # read all data
    X, y, users_train = read_train_txt(os.path.join(DATA_DIR, TRAIN_TXT))

    # X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=123)

    X_test, ids, users_test = read_test_txt(os.path.join(DATA_DIR, TEST_TXT))

    X_train, y_train = X, y

    # for i in range(len(X_train)):
    #     X_train[i] = " ".join(process_one_tweet(X_train[i]))

    # for i in range(len(X_dev)):
    #     X_dev[i] = " ".join(process_one_tweet(X_dev[i]))

    # for i in range(len(X_test)):
    #     X_test[i] = " ".join(process_one_tweet(X_test[i]))

    # merge data according to user id
    X_train2, y_train2 = get_user_dict(X_train, y_train, users_train)

    new_X_train = X_train + X_train2
    new_y_train = y_train + y_train2

    users_test, X_test, user_ids_dict = get_user_dict_test(X_test, ids, users_test)

    # # -------------- Stage 2: Tf-idf --------------

    # compute tf-idf features
    tokenizer = TweetTokenizer(preserve_case=False)
    vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize,
                                 sublinear_tf = True,
                                 ngram_range=(1,1))

    X_train = vectorizer.fit_transform(new_X_train)
    X_test =  vectorizer.transform(X_test)

    # X_dev =  vectorizer.transform(X_dev)

    print(X_train.shape)

    # # -------------- Stage 3: Training --------------

    print("--- Start training ---")

    svm = OneVsRestClassifier(LinearSVC(), n_jobs=-1)

    svm.fit(X_train, new_y_train)

    # nb = MultinomialNB()

    # nb.fit(X_train, y_train)

    # knn = KNeighborsClassifier()

    # knn.fit(X_train, y_train)

    print("--- finish training ---")

    # # -------------- Stage 4: Predictions --------------

    # print(svm.score(X_dev, y_dev))

    # print(nb.score(X_dev, y_dev))

    predictions = svm.predict(X_test)

    write_prediction(OUT_CSV, predictions, users_test, user_ids_dict)

    # # -------------- End --------------

if __name__ == '__main__':
    main()
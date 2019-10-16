import os
from utils import read_train_txt, read_test_txt, write_prediction, get_user_dict, get_user_dict_test
from preprocess import process_one_tweet, remove_test_users
import numpy as np

# for text processing
from sklearn.feature_extraction.text import TfidfVectorizer

# for machine learning
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

DATA_DIR = "data/"
DEV_TXT = "dev_tweets.txt"
TRAIN_TXT = "train_tweets.txt"
TEST_TXT = "test_tweets.txt"
OUT_CSV = "test_prediction.csv"

def main():

    # -------------- Stage 1: get user dict --------------

    # read all data
    X, y, users_train = read_train_txt(os.path.join(DATA_DIR, TRAIN_TXT))

    X_dev, y_dev, users_dev = read_train_txt(os.path.join(DATA_DIR, DEV_TXT))

    X_test, ids, users_test = read_test_txt(os.path.join(DATA_DIR, TEST_TXT))

    X_train, y_train = X, y

    # merge data according to user id
    X_train_merged, y_train_merged = get_user_dict(X_train, y_train, users_train)

    # merge dev data according to user id
    X_dev, y_dev = get_user_dict(X_dev, y_dev, users_dev)

    new_X_train = X_train + X_train_merged + X_dev
    new_y_train = y_train + y_train_merged + y_dev

    users_test, X_test, user_ids_dict = get_user_dict_test(X_test, ids, users_test)

    # # -------------- Stage 2: Tf-idf --------------

    # compute tf-idf features
    vectorizer = TfidfVectorizer(sublinear_tf = True,
                                 ngram_range=(1,1))

    
    X_test =  vectorizer.fit_transform(X_test)
    X_train = vectorizer.transform(new_X_train)
    X_dev = vectorizer.transform(X_dev)

    # # -------------- Stage 3: Training --------------

    print("--- Start training ---")

    svm = OneVsRestClassifier(LinearSVC(C = 1.5), n_jobs=-1)

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
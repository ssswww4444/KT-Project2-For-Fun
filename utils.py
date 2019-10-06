import csv
from collections import defaultdict, Counter
from preprocess import process_one_tweet

def read_train_txt(filename):
    with open(filename, "r", encoding="ISO-8859-1") as f:
        X = []
        y = []
        users = []
        for line in f:
            info = line[:-1].split(',')
            text = ",".join(info[2:-1])
            X.append(text[1:-1])
            y.append(info[-1])
            users.append(info[1])
    return X, y, users

def read_test_txt(filename):
    with open(filename, "r", encoding="ISO-8859-1") as f:
        X = []
        ids = []
        users = []
        for line in f:          
            info = line[:-1].split(',')
            text = ",".join(info[2:-1])
            X.append(text)
            ids.append(info[0])
            users.append(info[1])
    return X, ids, users

def write_prediction(outfile, predictions, users_test, user_ids_dict):
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        # header
        writer.writerow(["tweet-id", "class"])
        for i in range(len(predictions)):
            ids = user_ids_dict[users_test[i]]
            for user_id in ids:
                writer.writerow([user_id, predictions[i]])

def get_user_dict(X_train, y_train, users):

    # transform data into dict (user: text_ls)
    X = []
    y = []
    user_text_dict = defaultdict(list)
    user_loc_dict = {}

    for i in range(len(X_train)):
        text_ls = process_one_tweet(X_train[i])
        user_text_dict[users[i]] += text_ls
        user_loc_dict[users[i]] = y_train[i]

    for (key, value) in user_text_dict.items():
        X.append(" ".join(value))
        y.append(user_loc_dict[key])

    return X,y

def get_user_dict_test(X_test, ids, users):
    
    X = []
    user_text_dict = defaultdict(list)
    user_ids_dict = defaultdict(list)

    for i in range(len(X_test)):
        text_ls = process_one_tweet(X_test[i])
        user_text_dict[users[i]] += text_ls
        user_ids_dict[users[i]].append(ids[i])

    users = []
    for key, value in user_text_dict.items():
        users.append(key)
        X.append(" ".join(value))

    return users, X, user_ids_dict
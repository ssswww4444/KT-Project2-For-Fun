    # predictions = []
    # for test in X_test:
    #     scores = []
    #     for i in range(len(X)):
    #         scores.append(np.dot(test, X[i]))
    #     index = np.argmax(scores)
    #     predictions.append(y[index])
    # write_prediction(predictions)

    # return

    # # create inverted index
    # inverted_index = defaultdict(list)
    # count = 0
    # for i in range(len(X)):  # each user
    #     for j in range(len(X[i])):  # each term
    #         user = y[i]
    #         weight = X[i][j]
    #         term = vectorizer.get_feature_names()[j]
    #         inverted_index[term].append([user, weight])
    #     count += 1
    #     print("user: ", j)

    # with open("data/inverted_index.pickle", "wb") as f:
    #     pickle.dump(inverted_index, f)

    # return

    # predictions = []
    # count = 0

    # for test in X_test:
    #     scores = np.dot(X, test)
    #     # index = np.argmax(scores)
    #     # predictions.append(y[index])
    #     predictions.append(scores)
    #     count += 1
    #     if count % 30 == 0:
    #         print(count)

    # write_prediction(predictions)

    # y_pred = clf.predict(X_dev)

    # print(confusion_matrix(y_dev,y_pred))
    # print(classification_report(y_dev,y_pred))
    # print(accuracy_score(y_dev, y_pred))
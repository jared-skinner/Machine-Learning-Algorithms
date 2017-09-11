'''
Example: using logistic regression to classify handwriting digits from a



'''


from logistic_regression import LogisticRegression



# import data


# massage data


# get model
logistic = LogisticRegression(X, y, learning_rate=.1, number_of_epochs=1000)

# shuffle and split
logistic.shuffle_data()
logistic.split_data(percent_training=.7, percent_test=.3)

# train!
logistic.train_model()

# test


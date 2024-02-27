# Jade Oakes
# Final Project - Language ID
# 12/12/23

import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.util import bigrams
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')


def tokenize_features_unigrams(text):
    """Tokenize features into unigrams"""
    unigrams = {}
    for word in nltk.word_tokenize(text):
        unigrams[word] = 1.0

    return unigrams


def tokenize_features_characters(text):
    """Tokenize features into characters"""
    characters = {}
    for char in text:
        characters[char] = 1.0

    return characters


def tokenize_features_char_bigrams(text):
    """Tokenize features into character bigrams"""
    char_bigrams = {}
    char_bigrams_list = list(bigrams(text))
    for bigram in char_bigrams_list:
        char_bigrams[''.join(bigram)] = 1.0

    return char_bigrams


def create_features_list(train, dev, test, data, tokenize_features):
    """Create list of the dictionaries that contain the features"""
    train_features = train[data].apply(tokenize_features).tolist()
    dev_features = dev[data].apply(tokenize_features).tolist()
    test_features = test[data].apply(tokenize_features).tolist()

    return train_features, dev_features, test_features


def vectorize_features(train_features, dev_features, test_features):
    """Vectorize features using DictVectorizer()"""
    vectorizer = DictVectorizer()
    train_vector_features = vectorizer.fit_transform(train_features)
    dev_vector_features = vectorizer.transform(dev_features)
    test_vector_features = vectorizer.transform(test_features)

    return train_vector_features, dev_vector_features, test_vector_features


def naive_bayes(alpha, train_vector_features, eval_vector_features, train, eval_data, label, feature_name):
    """Multinomial Naive Bayes"""
    mn_naive_bayes = MultinomialNB(alpha=alpha)
    mn_naive_bayes.fit(train_vector_features, train[label])
    predictions = mn_naive_bayes.predict(eval_vector_features)
    # Results
    report = classification_report(eval_data[label], predictions, zero_division=1, digits=4)
    accuracy = accuracy_score(eval_data[label], predictions)
    print(f"Multimodal Naive Bayes report {feature_name}, alpha = {alpha}:\n", report)
    print(f"Multimodal Naive Bayes accuracy {feature_name}, alpha = {alpha}:\n", accuracy)


def naive_bayes_modified(alphas, train_vector_features, eval_vector_features, train, eval_data, label, feature_name):
    """Multinomial Naive Bayes"""
    for alpha in alphas:
        mn_naive_bayes = MultinomialNB(alpha=alpha)
        mn_naive_bayes.fit(train_vector_features, train[label])
        predictions = mn_naive_bayes.predict(eval_vector_features)
        # Results
        report = classification_report(eval_data[label], predictions, zero_division=1, digits=4)
        accuracy = accuracy_score(eval_data[label], predictions)
        print(f"Multimodal Naive Bayes report {feature_name}, alpha = {alpha}:\n", report)
        print(f"Multimodal Naive Bayes accuracy {feature_name}, alpha = {alpha}:\n", accuracy)


def logistic_regression(c, train_vector_features, eval_vector_features, train, eval_data, label, feature_name):
    log_regression = LogisticRegression(C=c, max_iter=1000)
    log_regression.fit(train_vector_features, train[label])
    predictions = log_regression.predict(eval_vector_features)
    # Results
    report = classification_report(eval_data[label], predictions, zero_division=1, digits=4)
    accuracy = accuracy_score(eval_data[label], predictions)
    print(f"Logistic Regression report {feature_name}, C = {c}:\n", report)
    print(f"Logistic Regression accuracy {feature_name}, C = {c}:\n", accuracy)


def logistic_regression_modified(c_vals, train_vector_features, eval_vector_features, train, eval_data, label, feature_name):
    """Multinomial Naive Bayes"""
    for c_val in c_vals:
        log_regression = LogisticRegression(max_iter=2000, C=c_val)
        log_regression.fit(train_vector_features, train[label])
        predictions = log_regression.predict(eval_vector_features)
        # Results
        report = classification_report(eval_data[label], predictions, zero_division=1, digits=4)
        accuracy = accuracy_score(eval_data[label], predictions)
        print(f"Logistic Regression report {feature_name}, C = {c_val}:\n", report)
        print(f"Logistic Regression accuracy {feature_name}, C = {c_val}:\n", accuracy)


def random_forest(n_estimators, max_depth, train_vector_features, eval_vector_features, train, eval_data, label, feature_name):
    rand_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rand_forest.fit(train_vector_features, train[label])
    predictions = rand_forest.predict(eval_vector_features)
    # Results
    report = classification_report(eval_data[label], predictions, zero_division=1, digits=4)
    accuracy = accuracy_score(eval_data[label], predictions)
    print(f"Random Forest report {feature_name}, n_estimator = {n_estimators}, max_depth = {max_depth}):\n", report)
    print(f"Random Forest accuracy {feature_name}, n_estimator = {n_estimators}, max_depth = {max_depth}):\n", accuracy)


def random_forest_modified1(n_estimators, train_vector_features, eval_vector_features, train, eval_data, label, feature_name):
    """Multinomial Naive Bayes"""
    for n_estimator in n_estimators:
        rand_forest = RandomForestClassifier(n_estimators=n_estimator, random_state=42)
        rand_forest.fit(train_vector_features, train[label])
        predictions = rand_forest.predict(eval_vector_features)
        # Results
        report = classification_report(eval_data[label], predictions, zero_division=1, digits=4)
        accuracy = accuracy_score(eval_data[label], predictions)
        print(f"Random Forest report {feature_name}, n_estimators = {n_estimator}:\n", report)
        print(f"Random Forest accuracy {feature_name}, n_estimators = {n_estimator}:\n", accuracy)


def random_forest_modified2(n_estimator, max_depths, train_vector_features, eval_vector_features, train, eval_data, label, feature_name):
    """Multinomial Naive Bayes"""
    for max_depth in max_depths:
        rand_forest = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=42)
        rand_forest.fit(train_vector_features, train[label])
        predictions = rand_forest.predict(eval_vector_features)
        # Results
        report = classification_report(eval_data[label], predictions, zero_division=1, digits=4)
        accuracy = accuracy_score(eval_data[label], predictions)
        print(f"Random Forest report {feature_name}, n_estimator = {n_estimator}, max_depth = {max_depth}):\n", report)
        print(f"Random Forest accuracy {feature_name}, n_estimator = {n_estimator}, max_depth = {max_depth}):\n", accuracy)


def main():
    # Load file
    file = pd.read_csv("dataset.csv")

    # Languages with Latin script
    selected_languages = ['Dutch', 'English', 'Estonian', 'French', 'Indonesian', 'Latin', 'Portuguese', 'Romanian', 'Spanish', 'Swedish', 'Turkish']
    # Select rows with the specified languages
    selected_data = file[file['language'].isin(selected_languages)]

    ### SPLIT DATASET

    # Split dataset into train and test_dev
    train, test_dev = train_test_split(selected_data, test_size=0.2, train_size=0.8, random_state=42)
    # Split dataset into dev and test
    test, dev = train_test_split(test_dev, test_size=0.5, train_size=0.5, random_state=42)

    ### CREATE FEATURES LISTS

    # Create unigram features list
    train_unigrams, dev_unigrams, test_unigrams = create_features_list(train, dev, test, "Text", tokenize_features_unigrams)
    # Create character features list
    train_characters, dev_characters, test_characters = create_features_list(train, dev, test, "Text", tokenize_features_characters)
    # Create character bigrams features list
    train_char_bigrams, dev_char_bigrams, test_char_bigrams = create_features_list(train, dev, test, "Text", tokenize_features_char_bigrams)

    ### VECTORIZE FEATURES

    # Vectorize unigrams
    train_vector_unigrams, dev_vector_unigrams, test_vector_unigrams = vectorize_features(train_unigrams, dev_unigrams, test_unigrams)
    # Vectorize characters
    train_vector_chars, dev_vector_chars, test_vector_chars = vectorize_features(train_characters, dev_characters, test_characters)
    # Vectorize character bigrams
    train_vector_char_bigrams, dev_vector_char_bigrams, test_vector_char_bigrams = vectorize_features(train_char_bigrams, dev_char_bigrams, test_char_bigrams)

    ### DEV SET EXPERIMENTATION

    # NAIVE BAYES
    print()

    # Hyperparameters
    alphas1 = np.arange(0.1, 1.1, 0.1)
    alphas2 = np.arange(1, 1100, 100)

    # Naive Bayes unigrams default
    naive_bayes(1.0, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    # Naive Bayes characters default
    naive_bayes(1.0, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    # Naive Bayes character bigrams default
    naive_bayes(1.0, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")

    # # Naive Bayes unigrams modified
    naive_bayes_modified(alphas1, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    naive_bayes_modified(alphas2, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    # Naive Bayes characters modified
    naive_bayes_modified(alphas1, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    naive_bayes_modified(alphas2, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    # Naive Bayes character bigrams modified
    naive_bayes_modified(alphas1, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")
    naive_bayes_modified(alphas2, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")

    # LOGISTIC REGRESSION
    print()

    # Hyperparameters
    c_vals1 = np.arange(1, 10, 1)
    c_vals2 = np.arange(0.1, 1.0, 0.1)

    # Logistic Regression unigrams default
    logistic_regression(1.0, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    # Logistic Regression characters default
    logistic_regression(1.0, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    # Logistic Regression character bigrams default
    logistic_regression(1.0, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")

    # Logistic Regression unigrams modified
    logistic_regression_modified(c_vals1, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    # Logistic Regression characters modified
    logistic_regression_modified(c_vals1,  train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    logistic_regression_modified(c_vals2, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    # Logistic Regression character bigrams modified
    logistic_regression_modified(c_vals1, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")
    logistic_regression_modified(c_vals2, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")

    # RANDOM FOREST
    print()

    # Hyperparameters
    n_estimators1 = np.arange(100, 600, 100)
    n_estimators2 = np.arange(10, 100, 10)
    max_depths1 = np.arange(1, 11, 1)
    max_depths2 = np.arange(10, 55, 5)

    # Random Forest unigrams default
    random_forest(100, None, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    # Random Forest characters default
    random_forest(100, None, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    # Random Forest character bigrams default
    random_forest(100, None, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")

    # Random Forest unigrams modified
    random_forest_modified1(n_estimators1, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    random_forest_modified1(n_estimators2, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "unigrams")
    random_forest_modified2(200, max_depths1, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "characters")
    random_forest_modified2(200, max_depths2, train_vector_unigrams, dev_vector_unigrams, train, dev, "language", "characters")
    # Random Forest characters modified
    random_forest_modified1(n_estimators1, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    random_forest_modified1(n_estimators2, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    random_forest_modified2(60, max_depths1, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    random_forest_modified2(60, max_depths2, train_vector_chars, dev_vector_chars, train, dev, "language", "characters")
    # Random Forest character bigrams modified
    random_forest_modified1(n_estimators1, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")
    random_forest_modified1(n_estimators2, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "characters")
    random_forest_modified2(60, max_depths1, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")
    random_forest_modified2(60, max_depths2, train_vector_char_bigrams, dev_vector_char_bigrams, train, dev, "language", "character bigrams")


    ### TEST SET RESULTS

    # Naive Bayes
    naive_bayes(0.1, train_vector_unigrams, test_vector_unigrams, train, test, "language", "unigrams")
    naive_bayes(0.9, train_vector_chars, test_vector_chars, train, test, "language", "characters")
    naive_bayes(0.8, train_vector_char_bigrams, test_vector_char_bigrams, train, test, "language", "character bigrams")

    # Logistic Regression
    logistic_regression(1.0, train_vector_unigrams, test_vector_unigrams, train, test, "language", "unigrams")
    logistic_regression(0.5, train_vector_chars, test_vector_chars, train, test, "language", "characters")
    logistic_regression(0.6, train_vector_char_bigrams, test_vector_char_bigrams, train, test, "language", "character bigrams")

    # Random Forest
    random_forest(200, 30, train_vector_unigrams, test_vector_unigrams, train, test, "language", "unigrams")
    random_forest(60, 20, train_vector_chars, test_vector_chars, train, test, "language", "characters")
    random_forest(60, 30, train_vector_char_bigrams, test_vector_char_bigrams, train, test, "language", "character bigrams")


if __name__ == '__main__':
    main()

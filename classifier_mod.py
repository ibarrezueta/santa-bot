"""
We use this module in cat_user.py because it uses pickled data created in
trained_classifier.py. This module contains functions that will classify a user
given their comment history
"""

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class VoteClassifier(ClassifierI):
    """
    Class in charge of classifying the data given different classification
    methods. Returns confidence of result as well
    """

    def __init__(self, *classifiers):
        """ Initializes with n number of classifiers"""
        self._classifiers = classifiers

    def classify(self, features):
        """ Classifies data """
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        """
        Returns confidence of given classification based off how uniform
        the vote of the classifiers was
        """
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# Loads variables from already compiled files

save_documents_f = open("load_files/documents.pickle","rb")
save_documents = pickle.load(save_documents_f)
save_documents_f.close()

all_words_f = open("load_files/all_words.pickle","rb")
all_words = pickle.load(all_words_f)
all_words_f.close()

word_features_f = open("load_files/word_features.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Loads 7 classifiers that have already been trained in train_classifiers.py

classifier_f = open("load_files/naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("load_files/MNB_classifier.pickle","rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("load_files/BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("load_files/LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("load_files/SGDClassifier_classifier.pickle","rb")
SGDClassifier_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("load_files/LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("load_files/NuSVC_classifier.pickle","rb")
NuSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

# initialization
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

def sentiment(text):
    """ Classifies data """
    feat = find_features(text)

    return voted_classifier.classify(feat), voted_classifier.confidence(feat)

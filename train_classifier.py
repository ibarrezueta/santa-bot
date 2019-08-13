"""
Here we train the classifies with data found in the load_files folder. We pass
labeled positive and labeled negative reviews and train 7 classifiers with those
sets of data. We then pickle the trained classifiers so that our main program
can load/run faster since every second counts when it's running live.
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

short_pos = open("reviews/pos_reviews.txt", "r").read()
short_neg = open("reviews/neg_reviews.txt", "r").read()

documents = []
all_words = []

allowed_word_types = ["J","R","V"]

for r in short_pos.split('\n'):
    documents.append( (r, "nice") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append( (r, "naughty") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_documents = open("load_files/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

save_all_words = open("load_files/all_words.pickle", "wb")
pickle.dump(all_words, save_all_words)
save_all_words.close()

save_word_features = open("load_files/word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featureset = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featureset)

training_set = featureset[:10000]

classifier = nltk.NaiveBayesClassifier.train(training_set)

save_classifier = open("load_files/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
save_classifier = open("load_files/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
save_classifier = open("load_files/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
save_classifier = open("load_files/LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
save_classifier = open("load_files/SGDClassifier_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
save_classifier = open("load_files/LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
save_classifier = open("load_files/NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

def sentiment(text):
    feat = find_features(text)

    return voted_classifier.classify(feat)

from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import svm
import io, re
import numpy as np

class WSD:
    """
    
    """
    def __init__(self, path='/Users/user/Desktop/EnglishLS.train'):
        self.path = path
        self.train_file = self.path + '/EnglishLS.train'
        self.key_file = self.path + '/EnglishLS.train.key'
        self.X_train = []
        self.Y_train = []

    def get_keys(self):
        """
        create dictionary of bnc : senseid(s)
        
        """
        sem_dict = {}
        with open(self.key_file, "r") as f:
            for line in f:
                line = line.rstrip().split(" ")
                if line[0] == 'appear.v':
                    if len(line) == 3:
                        sem_dict[line[1]] = (line[2],)
                    else:
                        sem_dict[line[1]] = (line[2], line[3])

        return sem_dict

    def get_instances(self):
        open_file = io.open(self.train_file, 'r').read()
        soup = bs(open_file, 'lxml')
        parent_tag = soup.find('lexelt', item='appear.v')

        text_doc = []
        answer_instance = []
        for children in parent_tag.find_all(id=re.compile("appear")):
            text_doc.append(children.get_text().strip())
            answer_instance.append(children.find_next(string=False).get('instance'))

        return text_doc, answer_instance

    def get_features(self):
        answer_keys = self.get_keys()
        documents, instances = self.get_instances()
        Y = []
        for i in instances:
            value = answer_keys.get(i)
            Y.append(value[0])

        vectorizer = CountVectorizer(min_df=1)
        X = vectorizer.fit_transform(documents)
        self.Y_train = Y
        self.X_train = X

    def get_trained_classifier(self, kernel='linear'):
        svm_classifier = svm.SVC(kernel=kernel, cache_size=1000)
        svm_classifier.fit(self.X_train, self.Y_train)
        return svm_classifier

    def get_score(self, classifier, k_fold=3):
        return cross_val_score(classifier, self.X_train, self.Y_train, cv=k_fold)

wsd = WSD()
wsd.get_features()
classifier = wsd.get_trained_classifier()
scores = wsd.get_score(classifier, 5)
print("Accuracy: %0.2f (+/

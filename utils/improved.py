from sklearn.linear_model import LogisticRegression
import nltk
from utils.dataset         import Dataset

from collections import Counter
import csv

pos_tags = {'NO_TAG':1, 'VERB':2, 'ADJ':3, 'PART':4, 'NOUN':5, 'PUNCT':6, 'NAMES':7, 'X':8, 'PROPN':9, 'SCONJ':10, 'ADV':11,
            'ADP':12, 'SYM':13, 'SPACE':14, 'NUM':15, 'CONJ':16, 'DET':17, 'AUX':18, 'PRON':19, 'CCONJ':20, 'INTJ':21, 'IDS':22}


# for word concreteness
class Concreteness(object):
    def __init__(self, filename='./utils/concreteness.csv'):
        self.data = {}
        with open(filename, 'r') as f:
            fCSV = csv.reader(f, delimiter=',', quotechar='"')
            next(fCSV)  # skip the header
            for row in fCSV:
                self.data[row[0].lower()] = float(row[2])

    def lookup(self, word):
        return self.data[word.lower()]


# for unigram frequency
class Frequency(object):
    def __init__(self, language):
        self.words = nltk.FreqDist()
        if language == 'english':
            from nltk.corpus import brown
            self.data = brown
            
            for sentence in self.data.sents():
                for word in sentence:
                    self.words[word.lower()] += 1
            
        elif language == 'spanish':
            trainData = Dataset(language)   
            trainData = trainData.trainset
            for word in trainData:
                word = word['target_word']
                self.words[word.lower()] += 1
        return None

    def lookup(self, word):
        return self.words.freq(word.lower())


# main class for CWI model
class Improved(object):
    def __init__(self, language, model=LogisticRegression, skipErrors=True, verbose=False,
                 features=['len_chars', 'len_tokens']):
        self.language   = language
        self.skipErrors = skipErrors
        self.verbose    = verbose
        self.features   = features
        self.frequency  = Frequency(language)

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if   language == 'english':
            self.avg_word_length = 5.3
            if 'syllables' in self.features:
                import pyphen
                self.syl = pyphen.Pyphen(lang='en')

            if 'concreteness' in self.features:
                self.concreteness = Concreteness()

            if 'pos' in features or 'embeddings' in self.features:
                import spacy
                self.nlp = spacy.load('en_core_web_lg')  # for word embeddings and pos tagging

            if 'frequency' in self.features:
                self.frequency = Frequency(language)

        elif language == 'spanish':
            self.avg_word_length = 6.2
            if 'syllables' in self.features:
                import pyphen
                self.syl = pyphen.Pyphen(lang='es')

            if 'pos' in self.features or 'embeddings' in self.features:
                import spacy
                self.nlp = spacy.load('es_core_news_md')  # for word embeddings and pos tagging
 
        self.model = model()


    def extract_features(self, sent):
        word     = sent['target_word']
        
        errors = []  # log our errors
        output = []  # store our list of features here
        
        if 'len_chars' in self.features:
            len_chars  = len(word) / self.avg_word_length
            output.append(len_chars)
        
        if 'len_tokens' in self.features:
            len_tokens = len(word.split())
            output.append(len_tokens)
        
        if 'syllables' in self.features:
            try:
                syllables = len(self.syl.inserted('copulate').split('-'))
                output.append(syllables)
            except KeyError:
                if self.verbose:
                    print('[syllables] not in dict:', word)
                output.append(10)
                errors.append(['syllables', word])

        if 'concreteness' in self.features:
            try:
                concreteness = self.concreteness.lookup(word)
                output.append(concreteness)
            except KeyError:
                if self.verbose:
                    print('[concreteness] not in dict:', word)                    
                output.append(0)
                errors.append(['concreteness', word])

        # for spacy 
        if 'pos' in self.features or 'embeddings' in self.features:
            we = self.nlp(word)

        if 'embeddings' in self.features:
            embedding = we.vector
            output += list(embedding)
        
        if 'pos' in self.features:
            pos = pos_tags[we[-1].pos_]
            output.append(pos)

        if 'frequency' in self.features:    
            try:
                frequency = self.frequency.lookup(word)
                if frequency == 0.:
                    raise KeyError
                output.append(frequency)
            except KeyError:
                if self.verbose:
                    print('[frequency] not in dict:', word)    
                output.append(0.)
                errors.append(['frequency', word])
        return output, errors


    def train(self, trainset):
        X = []
        y = []
        e = []
        for sent in trainset:
            feats, errors = self.extract_features(sent)
            e += errors
            if self.skipErrors and errors:
                continue
            X.append(feats)
            y.append(sent['gold_label'])
        
        # process errors
        print()
        print('Faulty cases:', len(e))
        print('Errors:', Counter([error[0] for error in e]))
        print()

        self.model.fit(X, y)


    def test(self, testset):
        X = []
        e = []
        for sent in testset:
            feats, errors = self.extract_features(sent)
            e += errors
            #if self.skipErrors and errors:
            #    continue
            X.append(feats)

        # process errors
        print()
        print('Faulty cases:', len(e))
        print('Errors:', Counter([error[0] for error in e]))
        print()

        return self.model.predict(X)



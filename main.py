# coding=latin-1
import os

import time
from pandas import DataFrame
import numpy
import tarfile
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


#inspired and partly copied from http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
NEWLINE = '\n'
SKIP_FILES = {'cmds'}

prepath ='C:/LokaleProjekter/scikitSpamFilter/'
def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    if '.tar.gz' in file_name:
                        tar = tarfile.open(file_path, "r:gz")
                        f = getFileFromTar(tar)
                    elif '.tar.bz2' in file_name:
                        past_header, lines = False, []
                        if '.tar.gz' in file_name:
                            tar = tarfile.open(file_path, "r:bz2")
                            f = getFileFromTar(tar)
                    else:
                        f = open(file_path)
                    for line in f:
                        if past_header:
                            lines.append(line.decode('latin-1'))
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def getFileFromTar(tar):
    for member in tar.getmembers():
        tfile = tar.extractfile(member)
        if tfile is not None:
            return tfile.read()


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    ('data/spam',        SPAM),
    #('data/easy_ham',    HAM),
    #('data/hard_ham',    HAM),
    ('data/beck-s',      HAM),
    ('data/farmer-d',    HAM),
    ('data/kaminski-v',  HAM),
    ('data/kitchen-l',   HAM),
    ('data/lokay-m',     HAM),
    #('data/williams-w3', HAM),
    ('data/BG',          SPAM)
   # ('data/GP',          SPAM)
   # ('data/SH',          SPAM)
]

print 'importing data'
data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(prepath+path, classification))

print 'reindiexing'
data = data.reindex(numpy.random.permutation(data.index))
examples = ['Free Viagra call today!', "I'm going to attend the Linux users group tomorrow.","The history was of Denmark"]

#manual,pipeline, Improoved, Improovedtfidf, Improoved_BernoulliNB, SVM?
mode = 'LinearSVC'


def folding(n_folds=6):
    global predictions
    k_fold = KFold(n=len(data), n_folds=n_folds)
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values

        #print 'fitting w cross validation,k_fold: '
        pipeline.fit(train_text, train_y)
        #print 'predicting w cross validation,k_fold: '
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label=SPAM)
        scores.append(score)
    print('Total documents classified:', len(data))
    print('Score:', sum(scores) / len(scores))
    print('Confusion matrix:')
    print(confusion)
    print(confusion/float(len(data)))


if mode=='manual':
    print 'generating word count'
    count_vectorizer = CountVectorizer()
    #fit_transform, does two things:
    #1) it learns the vocabulary of the corpus
    #2) then extracts word count features. how many occurances is there of each word in the corpus - the entire sample.
    counts = count_vectorizer.fit_transform(data['text'].values)

    # Use naive basian model to classify documents into given classes. It assumes each feature (here word count) is independent from every other one and contributes equally to the probability of
    # belonging to a class - not exaclty prefect, but should work just fine.
    # We call MultinomialNB and train it by calling fit

    print 'training'
    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(counts, targets)

    print 'playing around'
    #Now we have a 'spam' filter simply trained on the probability of being spam by content of words and how requent they appear in each class. Let's play:
    example_counts = count_vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print predictions # [1, 0]

elif mode=='pipeline':
    #now let's try to pipeline it! :
    print 'pipelining'

    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer()),
        ('classifier',  MultinomialNB()) ])

    # print 'pipeline fit'
    # pipeline.fit(data['text'].values, data['class'].values)
    # print 'pipeline predict'
    # print pipeline.predict(examples) # ['spam', 'ham']

    '''
    #Cross-Validating - to really challange3 our assumptions cross-validation of the model is essential. The simplest aproach would be taining/test split
    # - e.i into two sets. You can use up to 50% of and actualt dataset to train and stil get a valid result BUT there are alternatives that might be better suited.
    #k-fold cross-validation. Using this method, we split the data set into k parts, hold out one, combine the others and train on them,
    # then validate against the held-out portion. You repeat that process k times (each fold), holding out a different portion each time.
    #Then you average the score measured for each fold to get a more accurate estimation of your model's performance.
    '''

    t0 = time.time()
    folding()
    t1 = time.time()
    print 'Time to calc simple: ' + str(t1 - t0)

    #original blog results::
    # Total emails classified: 55326
    # Score: 0.942661080942
    # Confusion matrix: [[true 'spams', false 'spams'], [false 'hams', true 'hams']]
    # [[39%   0.3%]
    #   [ 6,3% 54%]]

    #scikit-learn provides various functions for evaluating the accuracy of a model. Here the F1 score is calculated (https://en.wikipedia.org/wiki/F1_score)for each fold,
    #  which we then average together for a mean accuracy on the entire set.
    # Using the model we just built and the example data sets mentioned in the beginning of this tutorial, we get about 0.94.
    #  A confusion matrix helps elucidate how the model did for individual classes. Out of 55,326 examples, we get about 178 false spams, and 3,473 false hams.
    # These numbers will vary each time we run the model.
    #('Total emails classified:', 26874)
    #('Score:', 0.94347628974807074) (from 1 pure true positive and true negative, to 0 only false neg/pos.)
    #Confusion matrix: [[true 'spams', false 'spams'], [false 'hams', true 'hams']]
    #[[16317    57]
    # [ 1072  9428]
    #[[60,7%    0,21%]
    # [ 3,99%  35,1%]
    #Time to calc simple: 79.4

elif mode=='Improoved':
    #one improovement is to look at n-grams - so sequences of words!
    print '2-gram'
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    t0 = time.time()
    folding()
    t1 = time.time()
    print 'Time to calc 1,2-grams: '+ str(t1-t0)
    '''
    skipped as memory chrashed:
    print '3-gram'
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 3))),
        ('classifier', MultinomialNB())
    ])
    #('Total emails classified:', 26874)
    #('Score:', 0.97572071158833829) vs 0.94 with BoW (simple wordcaounts above)
    #Confusion matrix:
    #[[16332    42]
    # [458 10042]]
    # [[60,1%    0,16%]    vs simple:   [[60,7%    0,21%]
    # [1,7% 37,4%]]                       [ 3,99%  35,1%]
    #Time to calc 1, 2 - grams: 268.92 a factor 3,4 slower than simple 1-gram

    t2 = time.time()
    folding()
    t3 = time.time()
    print 'Time to calc 1,2,3-grams: ' + str(t3 - t2)
    '''

elif mode=='Improovedtfidf':
    '''from http: // zacstewart.com / 2015 / 04 / 28 / document - classification - with-scikit - learn.html
    Another way to improve accuracy is to use different kinds of features. N-gram counts have the disadvantage of unfairly weighting longer documents.
    A six-word spammy message and a five-page, heartfelt letter with six "spammy" words could potentially receive the same "spamminess" probability.
    To counter that, we can use frequencies rather than occurances. That is, focusing on how much of the document is made up of a particular word,
    instead of how many times the word appears in the document. This kind of feature set is known as term frequencies.
    In addition to converting counts to frequencies, we can reduce noise in the features by reducing the weight for words that are common across the entire corpus.
    For example, words like "and," "the," and "but" probably don't contain a lot of information about the topic of the document,
    even though they will have high counts and frequencies across both ham and spam. To remedy that, we can use what's known as inverse document frequency or IDF.
    Adding another vectorizer to the pipeline will convert the term counts to term frequencies and apply the IDF transformation:
    '''

    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])

    t0 = time.time()
    folding()
    t1 = time.time()
    print 'Time to calc with bigram and tf-idf: ' + str(t1 - t0)
    # ('Total emails classified:', 26874)
    # ('Score:', 0.96320007585440848)
    # Confusion matrix:
    # [[16345    29]
    #  [718  9782]]
    # [[16345    0,11%]
    #  [2,67%  9782]]
    # Time to calc with tf - idf: 291.950000048

    '''So we got better false spams (contrary to tutorial on blog! but also got higher percentage of false hams.
    This might be good for spam filters, but the 'skewness' could be a problem in other classification tasks. '''

elif mode == 'Improoved_BernoulliNB':
        for ibin in [.999,100.,10000.]:
            print 'Binerize with: ' + str(ibin)
            pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
            ('classifier', BernoulliNB(binarize=ibin))])
            t0 = time.time()
            folding()
            t1 = time.time()
            print 'Time to calc with bigram and bernoulli classifier: (ibin='+str(ibin)+')' + str(t1 - t0)


        # Binerize with: 0.0 to 0.5 - no difference
        # ('Total emails classified:', 26874)
        # ('Score:', 0.91610686193067836)
        # Confusion matrix:
        # [[16367     7 (0,03%)]
        #  [ 1619 (6,02%)  8881]]
        #Time to calc with bigram and bernoulli classifier: (ibin=0.0) 286.1
        # Binerize with: 0.999
        # ('Total documents classified:', 26874)
        # ('Score:', 0.91588074457129409)
        # Confusion matrix:
        # [[16369     5]
        #  [ 1625  8875]]
        # [[  60.9%   0.0196%]
        #  [  6.05%   33.0%]]
        # Time to calc with bigram and bernoulli classifier: (ibin=0.999) 284.906999826
        #Binerize with: 10000.0
        # ('Total documents classified:', 26874)
        # ('Score:', 0.0)
        # Confusion matrix:
        # [[16374     0]
        #  [10500     0]]
        # Time to calc with bigram and bernoulli classifier: (ibin=10000.0) 292.142999887


elif mode=='SVM':
    #try with support vector machine as in http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,n_jobs=-1))])
    t0 = time.time()
    folding()
    t1 = time.time()
    print 'Time to calc with bigram w. td-idf and support vector machine classifier: ' + str(t1 - t0)
    # ('Total documents classified:', 26874)
    # ('Score:', 0.93709210984193936)
    # Confusion
    # matrix:
    # [[16338    36]
    #  [1210  9290]]
    # [[16338    36]
    #  [1210  9290]]
    #Time to calc with bigram w. td-idf and support vector machine classifier: 328.7
elif mode == 'LinearSVC':
    #like GDS above?
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', LinearSVC(penalty='l2', loss='hinge', tol=0.0001,random_state=42))])
    t0 = time.time()
    folding()
    t1 = time.time()
    print 'Time to calc with bigram w. td-idf and linear support vector  classifier: ' + str(t1 - t0)

    #('Total documents classified:', 26874)
    # ('Score:', 0.9940924278704234)
    # Confusion matrix:
    # [[16314    60]
    #  [   64 10436]]
    # [[ 0.60705515  0.00223264]
    #  [ 0.00238148  0.38833073]]
    # Time to calc with bigram w. td-idf and linear support vector classifier: 330.532000065

#this can be combined with a grid search of best parameters as desribed alter in the above link:
#from sklearn.model_selection import GridSearchCV
# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'clf__alpha': (1e-2, 1e-3),
# }
#gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
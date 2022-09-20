## Importing the necessary modules
import pandas as pd
import numpy as np
import pickle
import re
import os
from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK, space_eval, STATUS_FAIL
import hyperopt
from time import sleep
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import lightgbm as lgbm
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

def main():
    #############################################################
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    TRIALS_FOLDER = os.path.join(script_dir, 'trials')
    NUMBER_TRIALS_PER_RUN = 1
    #############################################################

    def get_texts(text_data):
        texts = [' '.join(text) for text in text_data]
        return [' '.join(text.split()) for text in texts]

    def add_zero_class(labels):
        augmented_labels = np.zeros((len(labels), len(labels[0]) + 1), dtype=np.int32)
        augmented_labels[:, :-1] = labels
        augmented_labels[:, -1] = (np.sum(labels, axis=1) == 0).astype('int32')
        return augmented_labels

    CONTRACTION_MAP = {"ain't": 'is not',"aren't": 'are not',"can't": 'cannot',"can't've": 'cannot have',"'cause": 'because',\
                   "could've": 'could have',"couldn't": 'could not',"couldn't've": 'could not have',"didn't": 'did not',\
                   "doesn't": 'does not',"don't": 'do not',"hadn't": 'had not',"hadn't've": 'had not have',"hasn't": 'has not',\
                   "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he'll've": 'he he will have',\
                   "he's": 'he is',"how'd": 'how did',"how'd'y": 'how do you',"how'll": 'how will',"how's": 'how is',\
                   "I'd": 'I would',"I'd've": 'I would have',"I'll": 'I will',"I'll've": 'I will have',"I'm": 'I am',\
                   "I've": 'I have',"i'd": 'i would',"i'd've": 'i would have',"i'll": 'i will',"i'll've": 'i will have',\
                   "i'm": 'i am',"i've": 'i have',"isn't": 'is not',"it'd": 'it would',"it'd've": 'it would have',"it'll": 'it will',\
                   "it'll've": 'it will have',"it's": 'it is',"let's": 'let us',"ma'am": 'madam',"mayn't": 'may not',\
                   "might've": 'might have',"mightn't": 'might not',"mightn't've": 'might not have',"must've": 'must have',\
                   "mustn't": 'must not',"mustn't've": 'must not have',"needn't": 'need not',"needn't've": 'need not have',\
                   "o'clock": 'of the clock',"oughtn't": 'ought not',"oughtn't've": 'ought not have',"shan't": 'shall not',\
                   "sha'n't": 'shall not',"shan't've": 'shall not have',"she'd": 'she would',"she'd've": 'she would have',\
                   "she'll": 'she will', "she'll've": 'she will have', "she's": 'she is', "should've": 'should have',\
                   "shouldn't": 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as',\
                   "that'd": 'that would', "that'd've": 'that would have', "that's": 'that is', "there'd": 'there would',\
                   "there'd've": 'there would have', "there's": 'there is', "they'd": 'they would', "they'd've": 'they would have',\
                   "they'll": 'they will', "they'll've": 'they will have', "they're": 'they are', "they've": 'they have',\
                   "to've": 'to have', "wasn't": 'was not', "we'd": 'we would', "we'd've": 'we would have', "we'll": 'we will',\
                   "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will',\
                   "what'll've": 'what will have', "what're": 'what are', "what's": 'what is', "what've": 'what have',\
                   "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where's": 'where is',\
                   "where've": 'where have', "who'll": 'who will', "who'll've": 'who will have', "who's": 'who is',\
                   "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not',\
                   "won't've": 'will not have', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have',\
                   "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are',\
                   "y'all've": 'you all have', "you'd": 'you would', "you'd've": 'you would have', "you'll": 'you will',\
                   "you'll've": 'you will have', "you're": 'you are', "you've": 'you have', 'gonna': 'going to', "which'll": 'which shall / which will',\
                   "everybody's": 'everybody is', "there're": 'there are', "who'd": 'who would / who had / who did', "dasn't": 'dare not',\
                   "'twas": 'it was', "o'er": 'over', "g'day": 'good day', "giv'n": 'given', "daresn't": 'dare not', "which's": 'which has / which is',\
                   "where're": 'where are', "shalln't": 'shall not', "e'er": 'ever', "how're": 'how are', "someone's": 'someone has / someone is',\
                   'gimme': 'give me', "somebody's": 'somebody has / somebody is', "'tis": 'it is', 'whilst': 'while still',\
                   "something's": 'something has / something is', "ne'er": 'never', "'s": 'is, has, does, or us', 'gotta': 'got to',\
                   "why're": 'why are', "I'm'o": 'I am going to', "noun're": 'noun are', "what'd": 'what did',\
                   "everyone's": 'everyone is', "this's": 'this has /', "so're": 'so are', "these're": 'these are',\
                   "those're": 'those are', "noun's": 'noun is', 'howdy': 'how do you do / how do you fare',\
                   'innit': 'is it not', 'methinks': 'me thinks', "gon't": 'go not', "ol'": 'old', "that'll": 'that shall / that will',\
                   "I'm'a": 'I am about to', "d'ye": 'do you / did you', "he've": 'he have', "may've": 'may have',\
                   'finna': 'fixing to / going to', "there'll": 'there shall / there will', "that're": 'that are',\
                   'wanna': 'want to', "who'd've": 'who would have',"amn't": 'am not',"daren't": 'dare not / dared not',\
                   "who're": 'who are',"why'd": 'why did','der': 'there'}

    contraction = sorted(CONTRACTION_MAP, key=len, reverse=True)
    c_re = re.compile('(%s)' % '|'.join(contraction))
    def expandContractions(text, c_re=c_re):
        def replace(match):
            return CONTRACTION_MAP[match.group(0)]
        return c_re.sub(replace, text.lower())

    link_regex = re.compile(r'(?:ftp|https?|www|file)\.?:?[//|\\\\]?[\w\d:#@%/;$()~_?\+-=\\\&]+\.[\w\d:#@%/;$~_?\+-=\\\&]+')

    dataset = load_dataset("lex_glue", "ecthr_a")

    train_text = get_texts(dataset['train']['text'])
    validation_text = get_texts(dataset['validation']['text'])
    test_text = get_texts(dataset['test']['text'])

    train_text = [  expandContractions(re.sub(link_regex,'',x).lower()) for x in train_text ]
    validation_text = [  expandContractions(re.sub(link_regex,'',x).lower()) for x in validation_text ]
    test_text = [  expandContractions(re.sub(link_regex,'',x).lower()) for x in test_text ]

    mlb = MultiLabelBinarizer(classes=range(10))
    mlb.fit(dataset['train']['labels'])

    train_labels = mlb.transform(dataset['train']['labels']).tolist()
    train_labels = add_zero_class(train_labels)

    validation_labels = mlb.transform(dataset['validation']['labels']).tolist()
    validation_labels = add_zero_class(validation_labels)

    test_labels = mlb.transform(dataset['test']['labels']).tolist()
    test_labels = add_zero_class(test_labels)

    def gb_mse_cv(params,random_state=42,train_x=train_text,train_y=train_labels,valid_x=validation_text,\
                                                                    valid_y=validation_labels,test_x=test_text,test_y=test_labels):

        new_params = {
              'boosting_type': params['boosting_type'],
              'num_leaves': int(params['num_leaves']),
              'learning_rate': params['learning_rate'],
              'subsample_for_bin': int(params['subsample_for_bin']),
              'feature_fraction': params['feature_fraction'],
              'bagging_fraction': params['bagging_fraction'],
              'min_data_in_leaf': int(params['min_data_in_leaf']),
              'lambda_l1': params['lambda_l1'],
              'lambda_l2': params['lambda_l2'],
              'min_child_weight': params['min_child_weight'],
              'n_estimators':int(params['n_estimators'])
             }

        if params['stopwords_type']:
            tfidf = tfidf = TfidfVectorizer(ngram_range=(1,3),min_df=10,stop_words=stopwords.words('english'))
        else:
            tfidf = TfidfVectorizer(ngram_range=(1,3),min_df=10,max_df=0.95)

        tfidf_vectorizer_vectors_train = tfidf.fit_transform(train_x)
        tfidf_vectorizer_vectors_validation = tfidf.transform(valid_x)
        tfidf_vectorizer_vectors_test = tfidf.transform(test_x)

        model = lgbm.LGBMClassifier(**new_params)

        if params['classifier_type'] == 'MultiOutputClassifier':
            multilabel_model = MultiOutputClassifier(model)
        elif params['classifier_type'] == 'ClassifierChain':
            multilabel_model = ClassifierChain(model)
        else:
            multilabel_model = OneVsRestClassifier(model)

        multilabel_model.fit(tfidf_vectorizer_vectors_train, train_y)

        train_pred = multilabel_model.predict(tfidf_vectorizer_vectors_train)
        train_score = (metrics.f1_score(train_y, train_pred, average="micro")\
                       + metrics.f1_score(train_y, train_pred, average="macro"))/2
        print("Train Score ",train_score)

        validation_pred = multilabel_model.predict(tfidf_vectorizer_vectors_validation)
        valid_score = (metrics.f1_score(valid_y, validation_pred, average="micro")\
                       + metrics.f1_score(valid_y, validation_pred, average="macro"))/2
        print("Valid Score ",valid_score)

        test_pred = multilabel_model.predict(tfidf_vectorizer_vectors_test)
        test_score = (metrics.f1_score(test_y, test_pred, average="micro") + \
                      metrics.f1_score(test_y, test_pred, average="macro"))/2
        print("Test  Score ",test_score)

        final_score = (valid_score + test_score)/2
        print("Final Score",final_score)

        return {'status': STATUS_OK,'loss': final_score}

    space={'boosting_type': hp.choice('boosting_type', ['gbdt','dart','goss']),
       'stopwords_type': hp.choice('stopwords_type', [True,False,]),
       'classifier_type': hp.choice('classifier_type', ['MultiOutputClassifier','ClassifierChain','OneVsRestClassifier']),
        'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        'min_child_weight': hp.loguniform('min_child_weight', -16, 5), #also aliases to min_sum_hessian
        'n_estimators': hp.quniform('n_estimators', 100, 10000, 1),
        }

    loaded_fnames = []
    trials = Trials()
    # Run new hyperparameter trials until killed
    while True:
        np.random.seed()
        # Load up all runs:
        for fname in os.listdir(TRIALS_FOLDER):
            num_fnmae = int(fname.split('.pkl')[0])
            if num_fnmae in loaded_fnames:
                continue
            print(os.path.join(TRIALS_FOLDER, fname))
            trials_obj = pickle.load(open(os.path.join(TRIALS_FOLDER, fname), 'rb'))

            n_trials = trials_obj['n']
            trials_obj = trials_obj['trials']
            sleep(0.1)
            trials.insert_trial_docs(trials_obj.trials[-n_trials:])
            trials.refresh()
            for i in range(len(trials.trials)):
                trials.trials[i]['tid'] = i
                trials.trials[i]['misc']['tid'] = i
                for key,val in trials.trials[i]['misc']['idxs'].items():
                    if len(val) != 0:
                        trials.trials[i]['misc']['idxs'][key] = [i]

            loaded_fnames.append(num_fnmae)

        if len(loaded_fnames) == 0:
            trials = Trials()

        n = NUMBER_TRIALS_PER_RUN
        print('^^^^^^^^^^^^^^^^^^')
        print("NEW TRILAS ",n + len(trials.trials))
        print('^^^^^^^^^^^^^^^^^^')
        best = None
        try:
            best=fmin(fn=gb_mse_cv, # function to optimize
                      space=space,
                      algo=anneal.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                      max_evals=n + len(trials.trials), # maximum number of iterations
                      trials=trials, # logging
                      verbose=1,
                      rstate=np.random.default_rng(os.getpid()) # fixing random state for the reproducibility
                     )

        except hyperopt.exceptions.AllTrialsFailed:
            pass

        print('current best', best)
        # Merge with empty trials dataset:
        TRIAL_NUMBER = 0 if len(loaded_fnames) == 0 else max(loaded_fnames) + 1 + os.getpid()
        pickle.dump({'trials': trials, 'n': n}, open(os.path.join(TRIALS_FOLDER, str(TRIAL_NUMBER) + '.pkl'), 'wb'))
        loaded_fnames.append(TRIAL_NUMBER)


if __name__ == "__main__":
    main()

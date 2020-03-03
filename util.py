import re
import os
import pandas as pd
import numpy as np
import pickle as pkl 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.utils import resample
from sklearn.utils import shuffle
from variables import  h5_file, bias, csv_path

user2cloth = {}
cloth2user = {}
usercloth2rating = {}
usercloth2rating_test = {}

def get_data():
    if not os.path.exists("train_upsample.csv") or not os.path.exists("test.csv"):
        print("Upsampling data !!!")
        df = pd.read_csv(csv_path)
        data = df[['Review Text','Recommended IND']]
        data['PreProcessed Text'] = data.apply(Add_dataframe_column, axis=1)
        upsample_data(data)
    train_data = pd.read_csv('train_upsample.csv')
    test_data  = pd.read_csv('test.csv')

    train_labels  = np.array(train_data['Recommended IND'],dtype=np.int32)
    test_labels   = np.array(test_data['Recommended IND'],dtype=np.int32)

    train_reviews = np.array(train_data['PreProcessed Text'],dtype='str')
    test_reviews  = np.array(test_data['PreProcessed Text'],dtype='str')
    print("Data is Ready!!!")
    return train_labels,test_labels,train_reviews,test_reviews

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    lemmatizer = WordNetLemmatizer() 
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def Add_dataframe_column(x):
    review = str(x['Review Text'])
    lemmatizer = WordNetLemmatizer() 
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def upsample_data(data):
    data_majority = data[data['Recommended IND'] == 1]
    data_minority = data[data['Recommended IND'] == 0]

    # bias = data_minority.shape[0]/data_majority.shape[0]

    # lets split train/test data first then 
    train = pd.concat([data_majority.sample(frac=0.8,random_state=200),
            data_minority.sample(frac=0.8,random_state=200)])
    test = pd.concat([data_majority.drop(data_majority.sample(frac=0.8,random_state=200).index),
            data_minority.drop(data_minority.sample(frac=0.8,random_state=200).index)])

    train = shuffle(train)
    test = shuffle(test)

    print('positive data in training:',(train['Recommended IND'] == 1).sum())
    print('negative data in training:',(train['Recommended IND'] == 0).sum())
    print('positive data in test:',(test['Recommended IND'] == 1).sum())
    print('negative data in test:',(test['Recommended IND'] == 0).sum())

    # Separate majority and minority classes in training data for up sampling 
    data_majority = train[train['Recommended IND'] == 1]
    data_minority = train[train['Recommended IND'] == 0]

    print("majority class before upsample:",data_majority.shape)
    print("minority class before upsample:",data_minority.shape)

    # Upsample minority class
    data_minority_upsampled = resample(data_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples= data_majority.shape[0],    # to match majority class
                                    random_state=42) # reproducible results
    
    # Combine majority class with upsampled minority class
    train_data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    
    # Display new class counts
    print("After upsampling\n",train_data_upsampled['Recommended IND'].value_counts(),sep = "")
    train_data_upsampled = shuffle(train_data_upsampled)

    train_data_upsampled = train_data_upsampled.dropna(axis = 0, how ='any')
    test = test.dropna(axis = 0, how ='any')
    train_data_upsampled.to_csv('train_upsample.csv', encoding='utf-8', index=False)
    test.to_csv('test.csv', encoding='utf-8', index=False)

def preprocessed_data(reviews):
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)

def balance_test_data(reviews,labels):
    positive_labels = labels[labels == 1]
    positive_reviews = reviews[labels == 1]

    negative_labels = labels[labels == 0]
    negative_reviews = reviews[labels == 0]

    minority_count = len(negative_reviews)
    majority_count = len(positive_reviews)

    idxs = np.random.randint(0, majority_count, minority_count)

    reviews1 = positive_reviews[idxs]
    labels1 = positive_labels[idxs]

    reviews , labels = shuffle(np.concatenate((reviews1,negative_reviews)),np.concatenate((labels1,negative_labels)))
    return reviews , labels

def get_update_data():
    df = pd.read_csv(csv_path)
    data = df[['ID','Clothing ID','Rating']]

    data = shuffle(data)
    train_set = int(0.8 * len(df))
    df_train = data.iloc[:train_set]
    df_test  = data.iloc[train_set:]

    def update_dictionaries(row):
        i = int(row['ID'])
        j = int(row['Clothing ID'])
        r = float(row['Rating'])

        if i not in user2cloth:
            user2cloth[i] = [j]
        else:
            user2cloth[i].append(j)

        if j not in cloth2user:
            cloth2user[j] = [i]
        else:
            cloth2user[j].append(i)

        usercloth2rating[(i,j)] = r

    def update_test_data(row):
        i = int(row['ID'])
        j = int(row['Clothing ID'])
        r = float(row['Rating'])
        usercloth2rating_test[(i,j)] = r

    df_train.apply(update_dictionaries, axis=1)
    df_test.apply(update_test_data, axis=1)


    with open('user2cloth.json', 'wb') as f:
        pkl.dump(user2cloth, f)

    with open('cloth2user.json', 'wb') as f:
        pkl.dump(cloth2user, f)

    with open('usercloth2rating.json', 'wb') as f:
        pkl.dump(usercloth2rating, f)

    with open('usercloth2rating_test.json', 'wb') as f:
        pkl.dump(usercloth2rating_test, f)


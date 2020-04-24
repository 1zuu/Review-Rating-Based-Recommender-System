import os
alpha = 0.8
#Sentiment analysis data
seed = 42
vocab_size = 15000
max_length = 120
embedding_dim = 512
trunc_type = 'post'
oov_tok = "<OOV>"
num_epochs = 5
batch_size = 128
size_lstm  = 128
size_dense = 64
size_output = 1
bias = 0.21600911256083669

#Recommender System data
cloth_count_threshold = 25
max_neighbor = 5

#Data paths and weights
train_data_path = 'data/train.csv'
test_data_path = 'data/test.csv'
sentiment_path = "data/model.json"
sentiment_weights = "data/model.h5"
eclothing_data = 'data/Womens Clothing E-Commerce Reviews.csv'
preprocessed_sentiment_data = "data/Preprocessed Sentiment Data.csv"
preprocessed_recommender_data = "data/Preprocessed Recommdender Data.csv"
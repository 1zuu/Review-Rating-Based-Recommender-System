import os
alpha = 0.7
table_name = 'ecloths'
# default_table_name =
db_url = 'mysql://root:Isuru767922513@localhost/bellarena'
#Sentiment analysis data
seed = 42
vocab_size = 15000
max_length = 120
embedding_dimS = 512
trunc_type = 'post'
oov_tok = "<OOV>"
num_epochs = 2
batch_size = 128
size_lstm  = 256
denseS = 64
size_output = 1
validation_split = 0.15
bias = 0.21600911256083669
sentiment_data = "data/sentiment_data"
sentiment_weights = "data/sentiment_data/sentiment_model.h5"

#Recommender System data
max_recommendes = 10
cutoff = 0.8
lr = 0.08
mom = 0.9
cloth_count_threshold = 25
embedding_dimR = 128
denseR = 512
R_hidden = 64
R_out = 1
batch_sizeR = 128
num_epochsR = 20
recommendation_data = "data/recommendation_data"
recommender_weights = "data/recommendation_data/recommender_weights_{}.h5"

#Data paths and weights
eclothing_data = 'data/Womens Clothing E-Commerce Reviews.csv'
import os

vocab_size = 15000
max_length = 120
embedding_dim = 300
trunc_type = 'post'
oov_tok = "<OOV>"
num_epochs = 10
sentiment_path = "model.json"
sentiment_weights = "model.h5"
batch_size = 128
size_lstm1 = 128
size_lstm2 = 64
size_dense = 64
size_output = 1
bias = 0.21600911256083669
preprocessed_path = "preprocessedEclothing.csv"
csv_path = 'Womens Clothing E-Commerce Reviews.csv' if not os.path.exists(preprocessed_path) else preprocessed_path
seed = 42
neighbours = 20
cloth_limit = 3

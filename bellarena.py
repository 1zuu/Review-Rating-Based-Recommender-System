from sentiment_analyser import SentimentAnalyser
from util import balance_test_data, get_data
from variables import bias,h5_file
import os

current_dir = os.getcwd()
h5_path = os.path.join(current_dir,h5_file)

if __name__ == "__main__":
    train_labels,test_labels,train_reviews,test_reviews = get_data()
    analyser = SentimentAnalyser(train_reviews,train_labels,test_reviews,test_labels)
    analyser.tokenizing_data()
    if os.path.exists(h5_file):
        print("Loading existing model !!!")
        analyser.load_model()
    else:
        print("Training the model  and saving!!!")
        analyser.embedding_model()
        analyser.train_model(bias)
        analyser.save_model()

    reviews,labels = balance_test_data(test_reviews,test_labels)
    analyser.predict(reviews[np.random.choice(len(reviews))],labels[np.random.choice(len(reviews))])
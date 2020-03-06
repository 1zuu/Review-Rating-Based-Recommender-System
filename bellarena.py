from sentiment_analyser import SentimentAnalyser
from util import balance_test_data, get_data, get_reviews_for_id
from variables import bias,sentiment_path,sentiment_weights
import os

current_dir = os.getcwd()
sentiment_path = os.path.join(current_dir,sentiment_path)
sentiment_weights = os.path.join(current_dir,sentiment_weights)

if __name__ == "__main__":
    train_labels,test_labels,train_reviews,test_reviews = get_data()
    analyser = SentimentAnalyser(train_reviews,train_labels,test_reviews,test_labels)
    analyser.tokenizing_data()
    if os.path.exists(sentiment_path) and os.path.exists(sentiment_weights):
        print("Loading existing model !!!")
        analyser.load_model()
    else:
        print("Training the model  and saving!!!")
        analyser.embedding_model()
        analyser.train_model(bias)
        analyser.save_model()

    # reviews,labels = balance_test_data(test_reviews,test_labels)
    # analyser.predict(reviews[np.random.choice(len(reviews))],labels[np.random.choice(len(reviews))])
    reviews, labels = get_reviews_for_id(767)
    analyser.predict(reviews[0],labels)
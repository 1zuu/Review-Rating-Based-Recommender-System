import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentiment_analyser import SentimentAnalyser
import logging
logging.getLogger('tensorflow').disabled = True
from recommender_system import RecommenderSystem
from util import get_sentiment_data, get_reviews_for_id, get_user_id, get_final_score
from variables import bias,sentiment_path,sentiment_weights, seed

current_dir = os.getcwd()
sentiment_path = os.path.join(current_dir,sentiment_path)
sentiment_weights = os.path.join(current_dir,sentiment_weights)

'''
python -W ignore bellarena.py

'''

if __name__ == "__main__":
    train_labels,test_labels,train_reviews,test_reviews = get_sentiment_data()
    # recommender system
    recommendations = RecommenderSystem()
    user_id = get_user_id()
    recommender_scores, rec_cloth_ids = recommendations.get_recommendation(user_id)

    # sentiment analysis
    analyser = SentimentAnalyser(train_reviews,train_labels,test_reviews,test_labels)
    analyser.run()
    sentiment_scores = analyser.predict_sentiments(rec_cloth_ids)
    # Final score
    get_final_score(recommender_scores, sentiment_scores, rec_cloth_ids)
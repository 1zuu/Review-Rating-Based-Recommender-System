import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentiment_analyser import SentimentAnalyser
import logging
logging.getLogger('tensorflow').disabled = True
from recommender_system import RecommenderSystem
from util import get_sentiment_data, get_reviews_for_id, get_user_id, get_final_score, get_recommendation_data
from variables import bias,sentiment_path,sentiment_weights, seed

current_dir = os.getcwd()
sentiment_path = os.path.join(current_dir,sentiment_path)
sentiment_weights = os.path.join(current_dir,sentiment_weights)

'''
python -W ignore bellarena.py

1187 users
147 cloths

'''

if __name__ == "__main__":
    # recommender system
    recommendations = RecommenderSystem()
    user_id = get_user_id()
    recommender_scores, rec_cloth_ids = recommendations.get_recommendation(user_id)

    # sentiment analysis
    analyser = SentimentAnalyser()
    analyser.run()
    sentiment_scores = analyser.predict_sentiments(rec_cloth_ids)
    # Final score
    get_final_score(recommender_scores, sentiment_scores, rec_cloth_ids)
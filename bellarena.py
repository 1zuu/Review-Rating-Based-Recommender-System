import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentiment_analyser import SentimentAnalyser
import logging
logging.getLogger('tensorflow').disabled = True
from mf import RecommenderSystem
from util import get_sentiment_data, get_reviews_for_id, get_user_id, get_final_score, get_recommendation_data, create_dataset
from variables import table_name, db_url
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
'''
python -W ignore bellarena.py

1187 users
147 cloths

'''
create_dataset()
data = pd.read_sql_table(table_name, db_url)
recommendations = RecommenderSystem(data)
recommendations.run()
analyser = SentimentAnalyser(data)
analyser.run()

def train_task():
    recommendations.run_finetune_mf()

if __name__ == "__main__":
    scheduler.add_job(func=train_task, trigger="interval", seconds=300)
    scheduler.start()

    user_id = get_user_id(data)
    rec_cloth_ids, recommender_scores  = recommendations.predict(user_id)
    sentiment_scores = analyser.predict_sentiments(rec_cloth_ids)

    # Final score
    recommended_ids, recommended_score = get_final_score(recommender_scores, sentiment_scores, rec_cloth_ids)
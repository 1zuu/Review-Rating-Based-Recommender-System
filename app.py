import os
import json
import pandas as pd

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentiment_analyser import SentimentAnalyser
import logging
logging.getLogger('tensorflow').disabled = True
from mf import RecommenderSystem
from variables import *
from util import get_final_score, create_dataset

from flask import Flask
from flask import jsonify
from flask import request
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()


'''
        python -W ignore app.py

'''

app = Flask(__name__)

create_dataset()
# default_data = pd.read_sql_table(table_name, db_url)

recommenders = RecommenderSystem()
recommenders.run()
sentiments = SentimentAnalyser()
sentiments.run()


def train_task():
    recommenders = RecommenderSystem()
    recommenders.run_finetune_mf()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    user_id = message['user_id']
    rec_cloth_ids, recommender_scores  = recommenders.predict(int(user_id))
    sentiment_scores = sentiments.predict_sentiments(rec_cloth_ids)
    recommended_ids, recommended_score = get_final_score(recommender_scores, sentiment_scores, rec_cloth_ids)

    response = {
            'input_id': recommended_ids
    }
    return jsonify(response)

if __name__ == "__main__":
    scheduler.add_job(func=train_task, trigger="interval", seconds=100)
    scheduler.start()
    app.run(debug=True, threaded=False)
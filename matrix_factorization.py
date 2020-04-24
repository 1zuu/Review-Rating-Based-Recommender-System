import os
import numpy as np
import pandas as pd
from variables import*
from util import get_recommendation_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate
from keras.models import Model, model_from_json
from keras.optimizers import SGD

import logging
logging.getLogger('tensorflow').disabled = True

class RecommenderSystem(object):
    def __init__(self):
        user_ids, cloth_ids,ratings = get_recommendation_data()
        self.user_ids = user_ids
        self.cloth_ids = cloth_ids
        self.ratings = ratings

        self.n_users = len(set(self.user_ids))
        self.n_cloths = len(set(self.cloth_ids))

    def split_data(self):
        Ntrain = int(cutoff * len(self.ratings))
        self.train_user_ids = self.user_ids[:Ntrain]
        self.train_cloth_ids = self.cloth_ids[:Ntrain]
        self.train_ratings = self.ratings[:Ntrain]

        self.test_user_ids = self.user_ids[Ntrain:]
        self.test_cloth_ids = self.cloth_ids[Ntrain:]
        self.test_ratings = self.ratings[Ntrain:]

        self.avg_rating = self.train_ratings.mean()
        self.std_rating = self.train_ratings.std()
        self.train_ratings = (self.train_ratings - self.avg_rating)/self.std_rating
        self.test_ratings  = (self.test_ratings - self.avg_rating)/self.std_rating

    def regressor(self):

        user_input = Input(shape=(1,))
        cloth_input = Input(shape=(1,))

        user_embedding = Embedding(self.n_users, embedding_dimR)(user_input)
        cloth_embedding = Embedding(self.n_cloths, embedding_dimR)(cloth_input)

        user_embedding = Flatten()(user_embedding)
        cloth_embedding = Flatten()(cloth_embedding)

        x = Concatenate()([user_embedding, cloth_embedding])
        # x = Dense(denseR, activation='relu')(x)
        x = Dense(R_hidden, activation='relu')(x)
        x = Dense(R_hidden, activation='relu')(x)
        x = Dense(R_hidden, activation='relu')(x)
        outputs = Dense(R_out, activation='relu')(x)

        model = Model(
            inputs=[user_input, cloth_input],
            outputs=outputs
            )

        self.model = model

    def train_model(self):
        self.model.compile(
                loss='mse',
                optimizer='adam')
                # optimizer=SGD(lr=lr, momentum=mom))
        self.model.summary()

        self.model.fit(
            x=[self.train_user_ids,self.train_cloth_ids],
            y=self.train_ratings,
            batch_size=batch_sizeR,
            epochs=num_epochsR,
            validation_data=(
                [self.test_user_ids,self.test_cloth_ids],
                self.test_ratings
                )
            )

    def save_model(self):
        model_json = self.model.to_json()
        with open(recommender_path, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(recommender_weights)

    def load_model(self):
        json_file = open(recommender_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(recommender_weights)
        loaded_model.compile(
                loss='mse',
                optimizer='adam')
                # optimizer=SGD(lr=lr, momentum=mom))
        self.model = loaded_model

    def run(self):
        self.split_data()
        if os.path.exists(recommender_path) and os.path.exists(recommender_weights):
            self.load_model()
        else:
            self.regressor()
            self.train_model()
            self.save_model()

    def predict(self, user_id):
            print(self.model.predict([user_id,[i]]))


if __name__ == "__main__":
    model = RecommenderSystem()
    model.run()
    model.predict([100],)
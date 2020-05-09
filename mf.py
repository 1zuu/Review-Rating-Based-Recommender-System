import os
import pickle
import numpy as np
import pandas as pd
from variables import*
from util import get_recommendation_data
from datetime import datetime
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Embedding, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import SGD

import logging
logging.getLogger('tensorflow').disabled = True


class RecommenderSystem(object):
    def __init__(self):
        data, user_ids, cloth_ids,ratings = get_recommendation_data()
        self.user_ids = user_ids
        self.cloth_ids = cloth_ids
        self.ratings = ratings
        self.data = data

        self.n_users = len(set(self.user_ids))
        self.n_cloths = len(set(self.cloth_ids))
        print("{} users and {} cloths".format(self.n_users, self.n_cloths))

    def split_data(self):
        current_time = str(time()).split('.')[0]
        self.recommender_weights = recommender_weights.format(current_time)

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
        x = Dense(R_hidden, activation='relu', name='dense1')(x)
        x = Dense(R_hidden, activation='relu', name='dense2')(x)
        x = Dense(R_hidden, activation='relu', name='dense3')(x)
        x = Dense(R_out, activation='relu', name='dense_out')(x)

        model = Model(
            inputs=[user_input, cloth_input],
            outputs=x
            )

        self.model = model

    def train_model(self):
        self.model.compile(
                loss='mse',
                optimizer='adam')
                # optimizer=SGD(lr=lr, momentum=mom))
        # self.model.summary()

        self.model.fit(
            [self.train_user_ids,self.train_cloth_ids],
            self.train_ratings,
            batch_size=batch_sizeR,
            epochs=num_epochsR,
            validation_data=(
                [self.test_user_ids,self.test_cloth_ids],
                self.test_ratings
                ),
            #verbose=0
            )

    def finetune_regressor(self):
        user_input = Input(shape=(1,))
        cloth_input = Input(shape=(1,))

        user_embedding = Embedding(self.n_users, embedding_dimR)(user_input)
        cloth_embedding = Embedding(self.n_cloths, embedding_dimR)(cloth_input)

        user_embedding = Flatten()(user_embedding)
        cloth_embedding = Flatten()(cloth_embedding)

        x = Concatenate()([user_embedding, cloth_embedding])

        for layer in self.model.layers[7:]:
            layer.trainable = False
            x = layer(x)

        model = Model(
            inputs=[user_input, cloth_input],
            outputs=x
            )
        self.model = model


    def save_model(self):
        self.model.save(self.recommender_weights)

    def load_model(self, weight_path=None):
        if weight_path:
            self.recommender_weights = weight_path
        loaded_model = load_model(self.recommender_weights)

        loaded_model.compile(
                loss='mse',
                optimizer='adam')
                # optimizer=SGD(lr=lr, momentum=mom))
        self.model = loaded_model
    def run(self):
        self.split_data()
        if len(os.listdir(recommendation_data)) >= 1:
            weight_path = os.path.join(recommendation_data, os.listdir(recommendation_data)[-1])
            self.load_model(weight_path)
        else:
            print("Training")
            self.regressor()
            self.train_model()
            self.save_model()

    def run_finetune_mf(self):
        print("Fine tuning")
        weight_path = os.path.join(recommendation_data, os.listdir(recommendation_data)[-1])
        self.load_model(weight_path)
        self.split_data()

        self.finetune_regressor()
        self.train_model()
        self.save_model()

    def predict(self, user_id):
        data = self.data
        cloth_ids = data['cloth_id'].values
        alread_rated_cloths = data[data['user_id'] == user_id]['cloth_id'].values
        cloth_ids = set(cloth_ids)
        rating_ids = []
        for cloth_id in cloth_ids:
            if cloth_id not in alread_rated_cloths:
                u =  np.array([user_id]).reshape(-1,1)
                c =  np.array([cloth_id]).reshape(-1,1)
                rating = float(self.model.predict([u,c]).squeeze())
                rating = (rating * self.std_rating) + self.avg_rating
                rating_ids.append((cloth_id, rating))
        rec_cloths = sorted(rating_ids,key=lambda x: x[1],reverse=True)[:max_recommendes]
        rec_cloth_ids = [v[0] for v in rec_cloths if v[1] > 0]
        rec_cloth_rating = [min(v[1], 5.0) for v in rec_cloths if v[1] > 0]
        return rec_cloth_ids, rec_cloth_rating

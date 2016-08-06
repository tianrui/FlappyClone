import tensorflow as tf
import keras
from keras.layers import *
import numpy as np
import yaml
import pdb
import csv

class Model(object):
    def __init__(self, input_dims):
        self.modfile = 'nnmodel'
        self.weightfile = 'weight'
        self.input_dims = input_dims
        #self.model = keras.models.Sequential([
        #    Dense(input_dims, input_dim=self.input_dims, activation='relu', init='uniform'),
        #    Dropout(0.5),
        #    Dense(8, activation='relu'),
        #    Dropout(0.5),
        #    Dense(4, activation='relu'),
        #    Dropout(0.5),
        #    Dense(1, activation='sigmoid')])
        self.n = 0
        self.val_score = 0.6
        return

    def save(self):
        self.arch = self.model.to_json()
        self.weights = self.model.get_weights()
        #pdb.set_trace()
        with open(self.modfile, 'wb+') as f:
            f.write(self.arch)
            f.close()
        self.model.save_weights(self.weightfile, overwrite=True)
        print 'Model arch and weights saved...'
        return
        
    def load(self):
        with open(self.modfile, 'rb') as f:
            self.arch = f.read()
        self.model = keras.models.model_from_json(self.arch)
        self.model.load_weights(self.weightfile)
        print 'Model arch and weights loaded...'
        return

    def train(self):
        #pdb.set_trace()
        in_layer = Input(shape=(self.input_dims,))
        dense_layer = Dense(self.input_dims/2, activation='relu', init='uniform')(in_layer)
        dense_layer = Dense(4, activation='relu')(dense_layer)
        preds = Dense(1, activation='sigmoid')(dense_layer)
        self.model = keras.models.Model(input=in_layer, output=preds)

        # data collection
        self.flaps = []
        self.noflaps = []
        self.crashflaps = []
        short_idx = [1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15]
        with open('flaplabels.csv', 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) < 16:
                    self.flaps.append([float(row_i) for row_i in row])
                else:
                    rowarr = np.asarray(row).astype(float)
                    self.flaps.append(list(rowarr[short_idx]))
        self.flaps = np.asarray(self.flaps)
        # If we didn't get more examples, we don't need to train
        if self.flaps.shape[0] > self.n:
            self.n = self.flaps.shape[0]
        else:
            return 0

        with open('crashlabels.csv', 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(self.noflaps) < len(self.flaps):
                    if len(row) < 16:
                        self.crashflaps.append([float(row_i) for row_i in row])
                    else:
                        rowarr = np.asarray(row).astype(float)
                        self.crashflaps.append(list(rowarr[short_idx]))
        self.crashflaps = np.asarray(self.crashflaps)

        with open('noflaplabels.csv', 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(self.noflaps) < len(self.flaps):
                    if len(row) < 16:
                        self.noflaps.append([float(row_i) for row_i in row]) 
                    else:
                        rowarr = np.asarray(row).astype(float)
                        self.noflaps.append(list(rowarr[short_idx]))
        self.noflaps = np.asarray(self.noflaps)
        
        # consolidate data
        n = self.flaps.shape[0] + self.noflaps.shape[0] + self.crashflaps.shape[0]
        data = np.zeros((self.flaps.shape[0]+self.noflaps.shape[0]+self.crashflaps.shape[0], 13))
        data[:self.flaps.shape[0], :-1] = self.flaps
        data[:self.flaps.shape[0], -1] = 1
        data[self.flaps.shape[0]:self.flaps.shape[0]+self.noflaps.shape[0], :-1] = self.noflaps
        data[self.flaps.shape[0]:self.flaps.shape[0]+self.noflaps.shape[0], -1] = 0
        data[self.flaps.shape[0]+self.noflaps.shape[0]:n, :-1] = self.crashflaps
        data[self.flaps.shape[0]+self.noflaps.shape[0]:n, -1] = 0
        np.random.shuffle(data)
        
        startidx = 0
        k = 10
        self.trainx = data[:0.8*n, :-1]
        self.trainy = data[:0.8*n, -1]
        self.valx = data[0.8*n:, :-1]
        self.valy = data[0.8*n:, -1]

        # Set up training
        self.model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        self.model.fit(
                self.trainx, self.trainy,
                nb_epoch=1,
                batch_size=64)
        val_score = self.model.evaluate(
                self.valx, self.valy,
                batch_size=32)
        return val_score

    def infer(self, inputs):
        #pdb.set_trace()
        inarray = np.asarray(inputs)
        output = self.model.predict(inputs, batch_size=1)
        #print 'Decision to jump is ', output
        return output

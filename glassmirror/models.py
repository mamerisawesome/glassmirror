#!/usr/bin/env python
# Darwin Bautista
# HomographyNet, from https://arxiv.org/pdf/1606.03798.pdf

import os.path

from tensorflow.keras.applications import MobileNet
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate

def create_model():
    model = Sequential(name='homographynet')
    model.add(InputLayer((120, 120, 3), name='input_1'))

    # 4 Layers with 64 filters, then another 4 with 120 filters
    filters = 4 * [3] + 4 * [120]
    for i, f in enumerate(filters, 1):
        model.add(Conv2D(f, 3, padding='same', activation='relu', name='conv2d_{}'.format(i)))
        model.add(BatchNormalization(name='batch_normalization_{}'.format(i)))
        # MaxPooling after every 2 Conv layers except the last one
        if i % 2 == 0 and i != 8:
            model.add(MaxPooling2D(strides=(2, 2), name='max_pooling2d_{}'.format(int(i/2))))

    model.add(Flatten(name='flatten_1'))
    model.add(Dropout(0.5, name='dropout_1'))
    model.add(Dense(120, activation='relu', name='dense_1'))
    model.add(Dropout(0.5, name='dropout_2'))

    # Regression model
    model.add(Dense(8, name='dense_2'))

    return model
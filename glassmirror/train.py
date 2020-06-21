import os.path
import sys
import math

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

from .callbacks import LearningRateScheduler
from .losses import mean_corner_error
from .models import create_model

DATA_BASE_DIR = "./data"
DATA_INPUT_DIR_PATH = "{}/pokemon-output/".format(DATA_BASE_DIR)
DATA_INPUT_COMPRESSED = "{}/dataset.npz".format(DATA_BASE_DIR)


def load_compressed_dataset():
    from random import shuffle
    data = np.load(DATA_INPUT_COMPRESSED)
    img_indices = [idx for idx in range(0, len(data["original"]))]
    # img_indices = [idx for idx in range(0, 10)]
    shuffle(img_indices)

    # shuffle based on generated array indices
    original = []
    warped = []
    offsets = []
    for img_idx in img_indices:
        original += [data["original"][img_idx]]
        warped += [data["warped"][img_idx]]
        offsets += [data["offsets"][img_idx]]

    # resource cleaning
    del data
    del img_indices

    return [
        offsets,
        warped,
        offsets,
    ]


def train():
    [original, warped, offsets] = load_compressed_dataset()
    data_to_train = warped
    model = create_model()

    # Configuration
    batch_size = 12 # from 64
    target_iterations = 160 # from 90,000 iterations
    base_lr = 0.1 #  from 0.005

    sgd = SGD(lr=base_lr, momentum=0.9)

    model.compile(optimizer=sgd, loss="mean_squared_error", metrics=[mean_corner_error])
    model.summary()

    save_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = ModelCheckpoint(os.path.join(save_path, "model.{epoch:02d}.h5"))

    # LR scaling as described in the paper: https://arxiv.org/pdf/1606.03798.pdf
    lr_scheduler = LearningRateScheduler(base_lr, 0.1, 50)

    # In the paper, the 90,000 iterations was for batch_size = 64
    # So scale appropriately
    target_iterations = int(target_iterations * 12 / batch_size)

    # Trainig: 80%
    # Testing: 20%
    TRAIN_SAMPLES_COUNT = round(len(data_to_train) * 0.8)
    TEST_SAMPLES_COUNT = round(len(data_to_train) * 0.2)

    # As stated in Keras docs
    steps_per_epoch = int(TRAIN_SAMPLES_COUNT / batch_size)
    epochs = int(math.ceil(target_iterations / steps_per_epoch))

    train_data = np.asarray(data_to_train[:TRAIN_SAMPLES_COUNT])
    x_labels = [i for i in range(0, TRAIN_SAMPLES_COUNT)]
    train_labels = np.asarray(x_labels)

    test_data = np.asarray(data_to_train[TRAIN_SAMPLES_COUNT:])
    y_labels = [i for i in range(0, TRAIN_SAMPLES_COUNT)]
    test_labels = np.asarray(y_labels)

    test_steps = int(TEST_SAMPLES_COUNT / batch_size)

    # Train
    model.fit(
        x=train_data,
        y=train_labels,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[lr_scheduler, checkpoint],
        validation_data=test_data,
        validation_steps=test_steps
    )

    # Step 1, make the training script work :)
    # Step 2, let me know if you get it working, the next for on the path will be open to you.

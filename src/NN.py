import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from numpy import ndarray
# noinspection PyUnresolvedReferences
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Softmax, Input, ReLU
from tensorflow.python.keras.losses import CategoricalCrossentropy

from Game import (
    Game,
    Position,
    augment_data,
)
from config import Config


def create_model(filters=128):
    state_input = Input(shape=(Game.board_height, Game.board_width, Game.num_layers))
    conv1 = ReLU()(Conv2D(filters, (3, 3), padding="same")(state_input))
    conv2 = ReLU()(Conv2D(filters, (3, 3), padding="same")(conv1))
    conv3 = ReLU()(Conv2D(filters, (3, 3), padding="same")(conv2))

    pol1 = ReLU()(Conv2D(32, (3, 3), padding="same", name="pol1")(conv3))
    pol2 = Flatten()(pol1)
    pol3 = ReLU()(Dense(128, name="pol3")(pol2))
    pol = Softmax()(Dense(Game.num_actions, name="pol_final")(pol3))

    val1 = ReLU()(Conv2D(32, (3, 3), padding="same")(conv3))
    flat = Flatten()(val1)
    val2 = ReLU()(Dense(128)(flat))
    val = Softmax()(Dense(3)(val2))

    model = Model(inputs=state_input, outputs=[pol, val])
    model.compile()
    return model


class NN:
    """A wrapper for the network"""

    def __init__(self, config: Config, file_path=None):
        self.loss = CategoricalCrossentropy()
        self.optimizer = Adam(config.learning_rate)
        if file_path is None:
            self.model = create_model()
        else:
            self.model = tf.keras.models.load_model(file_path)

    def update_config(self, config: Config):
        self.optimizer.learning_rate = config.learning_rate

    def policy_function(self, position: Position) -> Tuple[ndarray, ndarray]:
        """Evaluates the position and returns probabilities of actions and evaluation score"""
        state = position.vectorize()[np.newaxis, ...]
        act, val = self.model(state)
        return act.numpy()[0], val.numpy()[0]

    def dump(self, file_name: str = None, info: str = ""):
        if file_name is None:
            file_name = f"../models/model-{time.strftime('%Y%m%d_%H%M%S')}{f'_{info}' if info else ''}"
        self.model.save(file_name)

    @tf.function(reduce_retracing=True)
    def train_step(self, x, y_act, y_val):
        with tf.GradientTape() as tape:
            pred_act, pred_val = self.model(x)
            act_loss = self.loss(y_act, pred_act)
            val_loss = self.loss(y_val, pred_val)
            loss = act_loss + val_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train(self, config: Config, batch: Tuple[ndarray, ndarray, ndarray]):
        """Trains the NN on a batch of data collected from self-play"""
        x, y_act, y_val = augment_data(*batch)
        dataset = (
            tf.data.Dataset.from_tensor_slices((x, y_act, y_val))
                .shuffle(10000)
                .batch(config.batch_size)
        )
        for x, y_act, y_val in dataset:
            self.train_step(x, y_act, y_val)

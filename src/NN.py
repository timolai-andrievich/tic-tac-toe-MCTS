import numpy as np
from numpy import ndarray
from Game import (
    Position,
    Image,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    NUM_ACTIONS,
    NUM_LAYERS,
    position_from_image,
)
from typing import Tuple, List
import time
import tensorflow as tf
import tensorflow
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Softmax
from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


FILTERS = NUM_LAYERS * 32


class PolicyNN(Model):
    """Provides the interface for an neural network model"""

    def __init__(self):
        super(PolicyNN, self).__init__()

        self.conv0 = Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv1 = Conv2D(1, kernel_size=(3, 3), padding="same", activation="relu")
        self.flatten = Flatten(input_shape=(None, BOARD_HEIGHT, BOARD_WIDTH, 1))
        self.lin1 = Dense(
            NUM_ACTIONS,
            input_shape=(None, BOARD_HEIGHT * BOARD_WIDTH * 1),
            activation="softmax",
        )

    def call(self, input):
        input = self.conv0(input)
        input = self.conv1(input)
        input = self.flatten(input)
        input = self.lin1(input)
        return input


class ValueNN(Model):
    """Provides the interface for an neural network model"""

    def __init__(self):
        super(ValueNN, self).__init__()

        self.conv0 = Conv2D(
            16,
            kernel_size=(3, 3),
            padding="same",
            input_shape=(None, BOARD_HEIGHT, BOARD_WIDTH, NUM_LAYERS),
            activation="relu",
        )
        self.conv1 = Conv2D(
            1,
            kernel_size=(3, 3),
            padding="same",
            input_shape=(None, BOARD_HEIGHT, BOARD_WIDTH, 16),
            activation="relu",
        )
        self.flatten = Flatten()
        self.lin1 = Dense(3, activation="softmax")

    def call(self, input):
        input = self.conv0(input)
        input = self.conv1(input)
        input = self.flatten(input)
        input = self.lin1(input)
        return input


class NN:
    """A wrapper for the network"""

    def __init__(self, policy_file=None, value_file=None):
        self.loss = CategoricalCrossentropy()
        self.valueOptimizer = Adam()
        self.policyOptimizer = Adam()
        if policy_file:
            self.policyNN = tf.keras.models.load_model(policy_file)
        else:
            self.policyNN = PolicyNN()
        if value_file:
            self.valueNN = tf.keras.models.load_model(value_file)
        else:
            self.valueNN = ValueNN()

    def policy_function(self, position: Position) -> Tuple[ndarray, ndarray]:
        """Evaluates the position and returns probabilities of actions and evaluation score"""
        state = position.vectorize()[np.newaxis, ...]
        act = self.policyNN(state)
        val = self.valueNN(state)
        return act.numpy()[0], val.numpy()[0]

    def dump(self, file_name: str = None, info: str = ""):
        if file_name is None:
            file_name = f"../models/model-{time.strftime('%Y%m%d_%H%M%S')}{'_' if info else ''}{info}"
        self.valueNN.save(f"{file_name}_value.tf")
        self.policyNN.save(f"{file_name}_policy.tf")

    def train(self, batch: List[Tuple[Image, Tuple[ndarray, float]]]):
        """Trains the NN on a batch of data collected from self-play"""
        x = np.zeros((len(batch), BOARD_HEIGHT, BOARD_WIDTH, NUM_LAYERS))
        y_act = np.zeros((len(batch), NUM_ACTIONS))
        y_val = np.zeros((len(batch), 3))
        for i in range(len(batch)):
            img, (act, val) = batch[i]
            x[i] = position_from_image(img).vectorize()
            y_act[i] = act
            y_val[i] = val
        with tf.GradientTape() as act_tape, tf.GradientTape() as val_tape:
            pred_act, pred_val = self.policyNN(x), self.valueNN(x)
            act_loss = self.loss(y_act, pred_act)
            val_loss = self.loss(y_val, pred_val)
        act_gradients = act_tape.gradient(act_loss, self.policyNN.trainable_variables)
        val_gradients = val_tape.gradient(val_loss, self.valueNN.trainable_variables)
        self.policyOptimizer.apply_gradients(
            zip(act_gradients, self.policyNN.trainable_variables)
        )
        self.valueOptimizer.apply_gradients(
            zip(val_gradients, self.valueNN.trainable_variables)
        )

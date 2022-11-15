"""Contains classes related to neural networks used for position evaluation
"""
import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow.keras.optimizers import Adam  # Tensorflow is using lazy loaders pylint: disable=import-error disable=no-name-in-module
from tensorflow import keras  # pylint: disable=import-error disable=no-name-in-module
from tensorflow.keras import layers  # pylint: disable=import-error disable=no-name-in-module
from tensorflow.keras.losses import CategoricalCrossentropy  # pylint: disable=import-error disable=no-name-in-module

from game import (
    Game,
    Position,
    augment_data,
)
from config import Config


def conv_layer(inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Constructs a convolutional layer with ReLU activation and batch normalization.

    Args:
        inputs (tf.Tensor): Input tensor.
        filters (int): The number of filters in convolution.
        name (str): The name of the block.

    Returns:
        tf.Tensor: The output tensor of the block.
    """
    flow = inputs
    flow = layers.Conv2D(filters, (3, 3), padding="same",
                         name=f'{name}/conv')(flow)
    flow = layers.BatchNormalization(name=f'{name}/bn')(flow)
    flow = layers.ReLU(name=f'{name}/relu')(flow)
    return flow

def residual_layer(inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Constructs a residual layer with `filters` features.

    Args:
        inputs (tf.Tensor): Input tensor.
        filters (int): The number of filters in convolution.
        name (str): The name of the block.

    Returns:
        tf.Tensor: The output tensor of the block.
    """
    flow = inputs
    shortcut = flow
    flow = layers.Conv2D(filters, (3, 3), padding="same",
                         name=f'{name}/conv1')(flow)
    flow = layers.BatchNormalization(name=f'{name}/bn')(flow)
    flow = layers.ReLU(name=f'{name}/relu')(flow)
    flow = layers.Conv2D(filters, (3, 3), padding="same",
                         name=f'{name}/conv2')(flow)
    flow = layers.Add(name=f'{name}/add')([shortcut, flow])
    return flow


def create_model(filters=16) -> keras.Model:
    """Builds the neural network

    Args:
        filters (int, optional): The amount of filters in convolutional layers. Defaults to 128.

    Returns:
        keras.Model: Compiled keras model.
    """
    inputs = layers.Input(shape=(Game.board_height, Game.board_width,
                                      Game.num_layers))
    common = inputs
    common = conv_layer(common, filters, 'common/conv')
    common = residual_layer(common, filters, 'common/residual/1')
    common = residual_layer(common, filters, 'common/residual/2')

    pol = common
    pol = residual_layer(pol, filters, name="pol/residual/1")
    pol = conv_layer(pol, 1, name='pol/conv/1')
    pol = layers.Flatten(name='pol/flat')(pol)
    pol = layers.Softmax(name='pol/final/softmax')(pol)

    val = common
    val = residual_layer(val, filters, name='val/residual/1')
    val = layers.Flatten(name='val/flatten')(val)
    val = layers.Dense(128, name='val/dense')(val)
    val = layers.ReLU(name='val/dense/relu')(val)
    val = layers.Dense(3, name='val/final/dense')(val)
    val = layers.Softmax(name='val/final/softmax')(val)

    model = keras.Model(inputs, [pol, val])
    model.build(input_shape=(None, Game.board_height, Game.board_width,
                             Game.num_layers))
    return model


class Model:
    """A wrapper for the network"""

    def __init__(self, config: Config, file_path=None):
        self.loss = CategoricalCrossentropy()
        self.optimizer = Adam(config.learning_rate)
        if file_path is None:
            self.model = create_model()
        else:
            self.model = tf.keras.models.load_model(file_path)

    def update_config(self, config: Config):
        """Updates current model parameters from the config provided.

        Args:
            config (Config): New config.
        """
        self.optimizer.learning_rate = config.learning_rate

    def policy_function(self, position: Position) -> Tuple[ndarray, ndarray]:
        """Evaluates the position and returns probabilities of actions and outcome probabilities.

        Args:
            position (Position): Position to be evaluated.

        Returns:
            Tuple[ndarray, ndarray]: Action probabilities and outcome probabilities in format
            [Win, Draw, Lose].
        """
        state = position.get_state()[np.newaxis, ...]
        act, val = self.model(state)
        return act.numpy()[0], val.numpy()[0]

    def save(self, file_name: str = None, info: str = ""):
        """Saves the model's weights into a file. If no filename is provided, then
        ../models/model-%Y%m%d_%H%M%S is chosen, where %Y is current year (4 digits),
        %m - current month (2 digits), %d - current day (2 digits), %H - current hour
        (2 digits), %M - current minute (2 digits), %S - current second (2 digits).

        Args:
            file_name (str, optional): The path to where the weights should be saved.
            Defaults to None.
            info (str, optional): Additional info to be added to the end of the filename.
            Defaults to empty string.
        """
        if file_name is None:
            file_name = (f"../models/model-{time.strftime('%Y%m%d_%H%M%S')}"
                         f"{f'_{info}' if info else ''}")
        self.model.save(file_name)

    @tf.function(reduce_retracing=True)
    def train_step(self, states: ndarray, y_act: ndarray, y_val: ndarray):
        """Performs on training step on the model.

        Args:
            states (ndarray): A tensor with information about the positions.
            y_act (ndarray): A tensor with the action probabilities.
            y_val (ndarray): A tensor with outcome probabilities.
        """
        with tf.GradientTape() as tape:
            pred_act, pred_val = self.model(states)
            act_loss = self.loss(y_act, pred_act)
            val_loss = self.loss(y_val, pred_val)
            loss = act_loss + val_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

    def train(self, config: Config, batch: Tuple[ndarray, ndarray, ndarray]):
        """Trains the model on a given batch of data.

        Args:
            config (Config): Training parameters.
            batch (Tuple[ndarray, ndarray, ndarray]): The training data.
        """
        states, y_act, y_val = augment_data(*batch)
        dataset = (tf.data.Dataset.from_tensor_slices(
            (states, y_act, y_val)).shuffle(10000).batch(config.batch_size))
        for states, y_act, y_val in dataset:
            self.train_step(states, y_act, y_val)

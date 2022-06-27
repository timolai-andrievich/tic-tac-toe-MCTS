from io import BytesIO, TextIOWrapper

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
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim


class NNModel(nn.Module):
    """Provides the interface for an neural network model"""

    def __init__(self):
        super(NNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(12, 48, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=2, padding=1)

        self.val_conv1 = nn.Conv2d(96, 3, kernel_size=1)
        self.val_lin1 = nn.Linear(
            4 * 3 * BOARD_HEIGHT * BOARD_WIDTH, BOARD_HEIGHT * BOARD_WIDTH
        )
        self.val_lin2 = nn.Linear(BOARD_HEIGHT * BOARD_WIDTH, 1)

        self.act_conv1 = nn.Conv2d(96, 24, kernel_size=1)
        self.act_lin1 = nn.Linear(4 * 24 * BOARD_HEIGHT * BOARD_WIDTH, NUM_ACTIONS)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 3 * 4 * BOARD_HEIGHT * BOARD_WIDTH)
        x_val = F.relu(self.val_lin1(x_val))
        x_val = torch.tanh(self.val_lin2(x_val))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * 24 * BOARD_HEIGHT * BOARD_WIDTH)
        x_act = F.softmax(self.act_lin1(x_act), dim=1)

        return x_act, x_val[:, 0]


class NN:
    """A wrapper for the network"""

    def __init__(self, use_gpu=False, file=None):
        self._device = "cuda" if use_gpu else "cpu"
        if file:
            self._NN = NNModel().to(self._device)
            self._NN.load_state_dict(torch.load(file))
        else:
            self._NN = NNModel().to(self._device)
        self.optimizer = torch.optim.Adam(self._NN.parameters())

    def policy_function(self, position: Position) -> Tuple[ndarray, float]:
        """Evaluates the position and returns probabilities of actions and evaluation score"""
        input = torch.from_numpy(position.vectorize()).float().to(self._device)
        act, val = self._NN(input)
        return act.cpu().detach().numpy(), val[0]

    def dump(self, file_name=f"models/model-{time.strftime('%H:%M:%S_%d-%m-%Y')}.mod"):
        torch.save(self._NN.state_dict(), file_name)

    def train(self, batch: List[Tuple[Image, Tuple[ndarray, float]]]):
        """Trains the NN on a batch of data collected from self-play"""
        self.optimizer.zero_grad()
        x = np.zeros((len(batch), NUM_LAYERS, BOARD_HEIGHT, BOARD_WIDTH))
        y_act = np.zeros((len(batch), NUM_ACTIONS))
        y_val = np.zeros((len(batch)))
        for i in range(len(batch)):
            img, (act, val) = batch[i]
            x[i] = position_from_image(img).vectorize()
            y_act[i] = act
            y_val[i] = val
        x = torch.from_numpy(x).to(self._device).float()
        y_act = torch.from_numpy(y_act).to(self._device).float()
        y_val = torch.from_numpy(y_val).to(self._device).float()
        pred_act, pred_val = self._NN(x)
        val_loss = F.mse_loss(pred_val, y_val)
        act_loss = F.mse_loss(pred_act, y_act)
        loss = val_loss + act_loss
        loss.backward()
        self.optimizer.step()

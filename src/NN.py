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


FILTERS = NUM_LAYERS * 32
BLOCKS = 3

class NNModel(nn.Module):
    """Provides the interface for an neural network model"""

    def __init__(self, device):
        super(NNModel, self).__init__()

        self.conv0 = nn.Conv2d(NUM_LAYERS, FILTERS, kernel_size=(3, 3), padding=(1, 1))
        self.conv1 = nn.Conv2d(FILTERS, FILTERS, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(FILTERS, FILTERS, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(FILTERS, FILTERS, kernel_size=(3, 3), padding=(1, 1))

        self.val_conv1 = nn.Conv2d(FILTERS, NUM_LAYERS, kernel_size=(3, 3), padding=(1, 1))
        self.val_lin1 = nn.Linear(NUM_LAYERS * BOARD_HEIGHT * BOARD_WIDTH, 3)

        self.act_conv1 = nn.Conv2d(FILTERS, NUM_LAYERS, kernel_size=(3, 3), padding=(1, 1))
        self.act_lin1 = nn.Linear(NUM_LAYERS * BOARD_HEIGHT * BOARD_WIDTH, NUM_ACTIONS)

    def forward(self, input):
        x = F.relu(self.conv0(input))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, NUM_LAYERS * BOARD_HEIGHT * BOARD_WIDTH)
        x_val = F.softmax(self.val_lin1(x_val), dim=1)

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, NUM_LAYERS * BOARD_HEIGHT * BOARD_WIDTH)
        x_act = F.softmax(self.act_lin1(x_act), dim=1)

        return x_act, x_val


class NN:
    """A wrapper for the network"""

    def __init__(self, use_gpu=False, file=None):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if file:
            self._NN = NNModel(self._device).to(self._device)
            self._NN.load_state_dict(torch.load(file))
        else:
            self._NN = NNModel(self._device).to(self._device)
        self.optimizer = torch.optim.Adam(self._NN.parameters(), lr=2e-3, weight_decay=1e-4)

    def policy_function(self, position: Position) -> Tuple[ndarray, ndarray]:
        """Evaluates the position and returns probabilities of actions and evaluation score"""
        input = torch.from_numpy(position.vectorize()).float().to(self._device)
        act, val = self._NN(input)
        return act.cpu().detach().numpy()[0], val.cpu().detach().numpy()[0]

    def dump(self, file_name: str = None, info: str = ""):
        if file_name is None:
            file_name = f"../models/model-{time.strftime('%Y%m%d_%H%M%S')}_{info}.pt"
        torch.save(self._NN.state_dict(), file_name)

    def train(self, batch: List[Tuple[Image, Tuple[ndarray, float]]]):
        """Trains the NN on a batch of data collected from self-play"""
        self.optimizer.zero_grad()
        x = np.zeros((len(batch), NUM_LAYERS, BOARD_HEIGHT, BOARD_WIDTH))
        y_act = np.zeros((len(batch), NUM_ACTIONS))
        y_val = np.zeros((len(batch), 3))
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

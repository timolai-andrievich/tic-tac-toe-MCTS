"""Contains classes related to neural networks used for position evaluation
"""
import time
from typing import Tuple
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from game import (
    Game,
    augment_data,
)
from config import Config


class ConvLayer(nn.Module):
    """Convolutional block.
    Consists of convolution from `in_channels` to `out_channels`,
    batch normalization, and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes the convolution layer.
        Args:
            in_channels (int): The number of channels in the input tensor.
            out_channels (int): The number of channels the output tensor should have.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 'same')
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, inputs):
        """Forwards the inputs through the module and returns the result.
        Should not be called directly.
        """
        flow = inputs
        flow = self.conv(flow)
        flow = self.norm(flow)
        flow = self.act(flow)
        return flow


class ResidualLayer(nn.Module):
    """Residual block. Consists of:
        - Convolution from `channels` to `channels`
        - Normalization
        - ReLU activation
        - Second convolution from `channels` to `channels`
        - Residual skip connection
    """

    def __init__(self, channels: int):
        """Initializes the residual block with given number of channels.
        Args:
            channels (int): Amount of channels in input and output tensors.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 'same')
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 'same')

    def forward(self, inputs):
        """Forwards the inputs through the module and returns the result.
        Should not be called directly.
        """
        flow = inputs
        shortcut = flow
        flow = self.conv1(flow)
        flow = self.norm(flow)
        flow = self.act(flow)
        flow = self.conv2(flow)
        flow = shortcut + flow
        return flow


class Network(nn.Module):
    """The class that incapsulates the neural network that is used to evaluate the game states.
    """

    def __init__(self, filters: int = 128, blocks: int = 10) -> None:
        """Initializes a neural network with `blocks` residual blocks with `filters` channels each.
        Args:
            filters (int, optional): Filters in residual blocks. Defaults to 128.
            blocks (int, optional): Amount of residual blocks in common part. Defaults to 10.
        """
        super().__init__()
        self.init_conv = ConvLayer(Game.num_layers, filters)
        self.common = nn.Sequential(*([ResidualLayer(filters)] * blocks))
        self.pol = nn.Sequential(
            ConvLayer(filters, filters),
            nn.Conv2d(filters, 1, 1, 1, 'valid'),
            nn.Flatten(),
        )
        self.wdl = nn.Sequential(
            ConvLayer(filters, 8),
            nn.Flatten(),
            nn.Linear(Game.board_height * Game.board_width * 8, 128),
            nn.Linear(128, 3),
        )

    def forward(self, inputs, logits=False):
        """Forwards the inputs through the module and returns the result.
        Should not be called directly.
        """
        flow = inputs
        flow = flow.permute((0, 3, 1, 2))
        flow = self.init_conv(flow)
        flow = self.common(flow)
        pol = self.pol(flow)
        wdl = self.wdl(flow)
        if not logits:
            pol = torch.nn.functional.softmax(pol, dim=-1)
            wdl = torch.nn.functional.softmax(wdl, dim=-1)
        return [pol, wdl]


class Model:
    """The class that wraps the neural network and provides a policy function.
    Policy function is separated so the computations backend could be switched out more easily
    and to make the policy function compatible with numpy arrays.
    """

    def __init__(self,
                 config: Config,
                 file_path: str = None,
                 device: str = None):
        """Initializes the model and either tries to load parameters from the file
        specified in `file_path` or builds the network from scratch.

        Args:
            config (Config): Config containing hyperparameters for the model, such as learning rate.
            file_path (_type_, optional): File path to the network weights. Defaults to None.
            device (str, optional): Device onto which load the model. Defaults to cpu.
            Can be either 'cpu' or 'cuda'
        """
        self.loss = nn.MSELoss()
        if device is None:
            device = 'cpu'
        self.device = device
        if file_path is None:
            self.network = Network().to(self.device)
        else:
            self.network = Network().to(self.device)
            self.network.load_state_dict(torch.load(file_path))
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=config.learning_rate)

    def update_config(self, config: Config):
        """Updates current model hyperparameters from the config provided.

        Args:
            config (Config): The config with new hyperparameters.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config.learning_rate

    def policy_function(self, state: ndarray) -> Tuple[ndarray, ndarray]:
        """Evaluates the position and returns probabilities of actions and outcome probabilities.
        Args:
            position (Position): Position to be evaluated.
        Returns:
            Tuple[ndarray, ndarray]: Action probabilities and outcome probabilities in format
            [Win, Draw, Lose].
        """
        state = state[np.newaxis, ...].astype(np.float32)
        state = torch.from_numpy(state).to(self.device)  # pylint: disable=no-member
        act, val = self.network(state)  # pylint: disable=not-callable
        act = act.detach().cpu().numpy()
        val = val.detach().cpu().numpy()
        return (act[0], val[0])

    def save(self, file_name: str = None, info: str = ""):
        """Saves the model's weights into a file. If no filename is provided, then
        ../models/model-%Y%m%d_%H%M%S.pt is chosen, where %Y is current year (4 digits),
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
                         f"{f'_{info}' if info else ''}.pt")
        torch.save(self.network.state_dict(), file_name)

    def train_step(self, states: ndarray, y_act: ndarray, y_val: ndarray):
        """Performs on training step on the model.
        Args:
            states (ndarray): A tensor with information about the positions.
            y_act (ndarray): A tensor with the action probabilities.
            y_val (ndarray): A tensor with outcome probabilities.
        """
        states = states.to(self.device)
        y_act = y_act.to(self.device)
        y_val = y_val.to(self.device)
        pred_act, pred_val = self.network(states, logits=False)  # pylint: disable=not-callable
        act_loss = self.loss(pred_act, y_act)
        val_loss = self.loss(pred_val, y_val)
        loss = act_loss + val_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, config: Config, batch: Tuple[ndarray, ndarray, ndarray]):
        """Trains the model on a given batch of data.
        Args:
            config (Config): Training parameters.
            batch (Tuple[ndarray, ndarray, ndarray]): The training data.
        """
        states, y_act, y_val = augment_data(*batch)
        states, y_act, y_val = [
            states.astype(np.float32),
            y_act.astype(np.float32),
            y_val.astype(np.float32)
        ]
        dataset = TensorDataset(*map(torch.from_numpy, [states, y_act, y_val]))  # pylint: disable=no-member
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        for states, y_act, y_val in loader:
            self.train_step(states, y_act, y_val)

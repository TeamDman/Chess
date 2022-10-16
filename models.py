from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from definitions import Piece
from game import BOARD_LEN, BOARD_SHAPE, META_LEN, META_SHAPE, BoardMetadataTensorBatch, BoardTensorBatch, BoardTensor, State, QValueTensorBatch, StateTensorBatch

class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()

        self.conv1 = nn.LazyConv2d(out_channels=52, kernel_size=(3,3))
        self.batch1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(out_channels=32, kernel_size=(3,3))
        self.batch2 = nn.LazyBatchNorm2d()

        self.hidden1 = nn.LazyLinear(out_features=3000)
        self.hidden2 = nn.LazyLinear(out_features=100)

        self.out_board = nn.LazyLinear(out_features=BOARD_LEN)
        self.out_meta = nn.LazyLinear(out_features=META_LEN)

    def forward(
        self,
        state: StateTensorBatch,
    ) -> StateTensorBatch:
        x: torch.Tensor = self.conv1(state.board)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = torch.cat((
            x.flatten(start_dim=1),
            state.board.flatten(start_dim=1),
            state.metadata,
        ), dim=1)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = torch.cat((x, state.board.flatten(start_dim=1)), dim=1)
        x_board, x_meta = self.out_board(x), self.out_meta(x)
        x_board = x_board.reshape(BOARD_SHAPE)
        x_meta = x_meta.reshape(META_SHAPE)
        return StateTensorBatch(
            board=x_board,
            metadata=x_meta
        )


class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic, self).__init__()
        

        self.board_conv1 = nn.LazyConv2d(out_channels=52, kernel_size=(3,3))
        self.board_batch1 = nn.LazyBatchNorm2d()
        self.board_conv2 = nn.LazyConv2d(out_channels=32, kernel_size=(3,3))
        self.board_batch2 = nn.LazyBatchNorm2d()

        self.action_conv1 = nn.LazyConv2d(out_channels=52, kernel_size=(3,3))
        self.action_batch1 = nn.LazyBatchNorm2d()
        self.action_conv2 = nn.LazyConv2d(out_channels=32, kernel_size=(3,3))
        self.action_batch2 = nn.LazyBatchNorm2d()


        self.hidden1 = nn.LazyLinear(out_features=3000)
        self.hidden2 = nn.LazyLinear(out_features=100)

        self.out = nn.LazyLinear(out_features=1)

    def forward(
        self,
        state: StateTensorBatch,
        action: StateTensorBatch,
    ) -> QValueTensorBatch:
        x_board: Tensor = state.board
        x_board = self.board_conv1(x_board)
        x_board = self.board_batch1(x_board)
        x_board = F.relu(x_board)
        x_board = self.board_conv2(x_board)
        x_board = self.board_batch2(x_board)
        x_board = F.relu(x_board)
        x_board_meta: Tensor = state.metadata

        x_action: Tensor = action.board
        x_action= self.action_conv1(x_action)
        x_action = self.action_batch1(x_action)
        x_action = F.relu(x_action)
        x_action = self.action_conv2(x_action)
        x_action = self.action_batch2(x_action)
        x_action = F.relu(x_action)
        x_action_meta: Tensor = action.metadata

        latent: Tensor  = torch.cat((
            x_board.flatten(start_dim=1),
            x_board_meta,
            x_action.flatten(start_dim=1),
            x_action_meta,
        ), dim=1)

        latent = self.hidden1(latent)
        latent = F.relu(latent)
        latent = self.hidden2(latent)
        latent = F.relu(latent)
        q = self.out(latent)
        q = q.flatten()
        return q
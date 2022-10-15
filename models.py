from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from definitions import Piece
from game import BOARD_LEN, BOARD_SHAPE, META_LEN, META_SHAPE, BatchedBoardMetadataTensor, BatchedBoardTensor, BoardTensor, State, BatchedQValueTensor

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
        board: BatchedBoardTensor,
        meta: BatchedBoardMetadataTensor
    ) -> Tuple[BatchedBoardTensor, BatchedBoardMetadataTensor]:
        x: torch.Tensor = self.conv1(board)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = torch.cat((x.flatten(start_dim=1), board.flatten(start_dim=1)), dim=1)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = torch.cat((x, board.flatten(start_dim=1)), dim=1)
        x_board, x_meta = self.out_board(x), self.out_meta(x)
        x_board = x_board.reshape(BOARD_SHAPE)
        x_meta = x_meta.reshape(META_SHAPE)
        return x_board, x_meta


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
        board: BatchedBoardTensor,
        meta: BatchedBoardMetadataTensor,
        proto_actions: BatchedBoardTensor,
        proto_actions_meta: BatchedBoardMetadataTensor,
    ) -> BatchedQValueTensor:
        batch_size = proto_actions.shape[0]
        x_board: torch.Tensor = self.board_conv1(board)
        x_board = self.board_batch1(x_board)
        x_board = F.relu(x_board)
        x_board = self.board_conv2(x_board)
        x_board = self.board_batch2(x_board)
        x_board = F.relu(x_board)
        x_board = x_board.repeat((batch_size,1,1,1))
        meta = meta.repeat((batch_size,1))
        x_action: torch.Tensor = self.action_conv1(proto_actions)
        x_action = self.action_batch1(x_action)
        x_action = F.relu(x_action)
        x_action = self.action_conv2(x_action)
        x_action = self.action_batch2(x_action)
        x_action = F.relu(x_action)

        latent = torch.cat((
            x_board.flatten(start_dim=1),
            meta,
            x_action.flatten(start_dim=1),
            proto_actions_meta,
        ), dim=1)
        latent = self.hidden1(latent)
        latent = F.relu(latent)
        latent = self.hidden2(latent)
        latent = F.relu(latent)
        q = self.out(latent)
        q = q.flatten()
        return q
import torch
import torch.nn as nn
import torch.nn.functional as F
from definitions import Piece

from game import BatchedBoardTensor, BoardTensor, State

class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()

        self.conv1 = nn.LazyConv2d(out_channels=52, kernel_size=(3,3))
        self.batch1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(out_channels=32, kernel_size=(3,3))
        self.batch2 = nn.LazyBatchNorm2d()

        self.hidden1 = nn.LazyLinear(out_features=3000)
        self.hidden2 = nn.LazyLinear(out_features=100)

        self.out = nn.LazyLinear(out_features=8*8*len(Piece))

    def forward(self, board: BatchedBoardTensor) -> State:
        x: torch.Tensor = self.conv1(board)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.cat((x.flatten(start_dim=1), board.flatten(start_dim=1)), dim=1)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = torch.cat((x, board.flatten(start_dim=1)), dim=1)
        x = self.out(x)
        x = x.reshape((-1,8,8,len(Piece)))
        return x

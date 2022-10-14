from __future__ import annotations
from dataclasses import dataclass
import enum

from torch import Tensor
import torch

@enum.unique
class Piece(enum.Enum):
    AIR = 0
    PAWN = 1
    ROOK = 2
    KNIGHT = 3
    BISHOP = 4
    QUEEN = 5
    KING = 6


@dataclass(frozen=True)
class State:
    board: Tensor # shape=(8,8,len(pieces))
    black_left_rook_moved: bool = False
    black_right_rook_moved: bool = False
    black_king_moved: bool = False
    white_left_rook_moved: bool = False
    white_right_rook_moved: bool = False
    white_king_moved: bool = False

    @staticmethod
    def get_starting_state() -> State:
        board = torch.zeros(8,8,len(Piece))
        board[[1,6],:,Piece.PAWN.value] = 1
        board[[0,0,7,7],[0,7,0,7], Piece.ROOK.value] = 1 
        board[[0,0,7,7],[1,6,1,6], Piece.KNIGHT.value] = 1 
        board[[0,0,7,7],[2,5,2,5], Piece.BISHOP.value] = 1 
        board[[0,7],[3,3], Piece.QUEEN.value] = 1
        board[[0,7],[4,4], Piece.KING.value] = 1
        board[2:6,:,Piece.AIR.value] = 1
        # board[board==torch.zeros(7), Piece.AIR.value] = 12
        return State(board=board)

class Game:
    state: State

    def __init__(self) -> None:
        self.state = State.get_starting_state()

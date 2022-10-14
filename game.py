from __future__ import annotations
from dataclasses import dataclass
from enum import unique, auto, Enum

from torch import Tensor
import torch

@unique
class Piece(Enum):
    AIR = 0
    BLACK_PAWN = auto()
    BLACK_ROOK = auto()
    BLACK_KNIGHT = auto()
    BLACK_BISHOP = auto()
    BLACK_QUEEN = auto()
    BLACK_KING = auto()
    WHITE_PAWN = auto()
    WHITE_ROOK = auto()
    WHITE_KNIGHT = auto()
    WHITE_BISHOP = auto()
    WHITE_QUEEN = auto()
    WHITE_KING = auto()


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
        board[1,:,Piece.BLACK_PAWN.value] = 1
        board[6,:,Piece.WHITE_PAWN.value] = 1
        board[[0,0],[0,7], Piece.BLACK_ROOK.value] = 1 
        board[[7,7],[0,7], Piece.WHITE_ROOK.value] = 1 
        board[[0,0],[1,6], Piece.BLACK_KNIGHT.value] = 1 
        board[[7,7],[1,6], Piece.WHITE_KNIGHT.value] = 1 
        board[[0,0],[2,5], Piece.BLACK_BISHOP.value] = 1 
        board[[7,7],[2,5], Piece.WHITE_BISHOP.value] = 1 
        board[0,3, Piece.BLACK_QUEEN.value] = 1
        board[7,3, Piece.WHITE_QUEEN.value] = 1
        board[0,4, Piece.BLACK_KING.value] = 1
        board[7,4, Piece.WHITE_KING.value] = 1
        board[2:6,:,Piece.AIR.value] = 1
        # board[board==torch.zeros(7), Piece.AIR.value] = 12
        return State(board=board)

    def is_checkmate(self) -> bool:
        pass

class Game:
    state: State

    def __init__(self) -> None:
        self.state = State.get_starting_state()

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List
from torch import Tensor
import torch
from definitions import Piece, black_pieces, PlayerColour, white_pieces

def can_move(player: PlayerColour, piece: Piece) -> bool:
    if player == "white" \
    and piece.value >= Piece.WHITE_PAWN.value \
    and piece.value <= Piece.WHITE_KING.value:
        return True
    if player == "black" \
    and piece.value >= Piece.BLACK_PAWN.value \
    and piece.value <= Piece.BLACK_KING.value:
        return True
    return False

@dataclass(frozen=True)
class PiecePosition:
    row: int
    col: int
    piece: Piece

BoardTensor = Tensor # shape=(8,8,len(pieces))
@dataclass(frozen=True)
class State:
    board: BoardTensor
    black_left_rook_moved: bool = False
    black_right_rook_moved: bool = False
    black_king_moved: bool = False
    white_left_rook_moved: bool = False
    white_right_rook_moved: bool = False
    white_king_moved: bool = False

    @staticmethod
    def get_empty_state() -> State:
        return State(board=torch.zeros(8,8,len(Piece)))

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

    def with_pieces(self, pieces: List[PiecePosition]) -> State:
        new_board = self.board.clone()
        for piece in pieces:
            new_board[piece.row, piece.col, :] = 0
            new_board[piece.row, piece.col, piece.piece.value] = 1
        return replace(self, board=new_board)

    def get_piece(self, row: int, col: int) -> Piece:
        return Piece(int(self.board[row, col].argmax()))

    def get_valid_moves(self, player: PlayerColour) -> List[State]:
        rtn: List[State] = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                # WHITE PAWN ADVANCE
                if player == "white" \
                and piece == Piece.WHITE_PAWN \
                and row > 0 \
                and self.get_piece(row - 1, col) == Piece.AIR:
                    rtn.append(self.with_pieces([
                        PiecePosition(row,col,Piece.AIR),
                        PiecePosition(row - 1, col, Piece.WHITE_PAWN),
                    ]))
                # WHITE PAWN CAPTURE LEFT
                if player == "white" \
                and piece == Piece.WHITE_PAWN \
                and row > 0 \
                and col > 0 \
                and self.get_piece(row - 1, col - 1).colour == "black":
                    rtn.append(self.with_pieces([
                        PiecePosition(row,col,Piece.AIR),
                        PiecePosition(row - 1, col - 1, Piece.WHITE_PAWN),
                    ]))
                # WHITE PAWN CAPTURE RIGHT
                if player == "white" \
                and piece == Piece.WHITE_PAWN \
                and row > 0 \
                and col < 7 \
                and self.get_piece(row - 1, col + 1).colour == "black":
                    rtn.append(self.with_pieces([
                        PiecePosition(row,col,Piece.AIR),
                        PiecePosition(row - 1, col + 1, Piece.WHITE_PAWN),
                    ]))
                # BLACK PAWN ADVANCE
                if player == "black" \
                and piece == Piece.BLACK_PAWN \
                and row < 7 \
                and self.get_piece(row + 1, col) == Piece.AIR:
                    rtn.append(self.with_pieces([
                        PiecePosition(row,col,Piece.AIR),
                        PiecePosition(row + 1, col, Piece.BLACK_PAWN),
                    ]))
                # BLACK PAWN CAPTURE LEFT
                if player == "black" \
                and piece == Piece.BLACK_PAWN \
                and row < 7 \
                and col > 0 \
                and self.get_piece(row + 1, col - 1).colour == "white":
                    rtn.append(self.with_pieces([
                        PiecePosition(row,col,Piece.AIR),
                        PiecePosition(row + 1, col - 1, Piece.BLACK_PAWN),
                    ]))
                # BLACK PAWN CAPTURE RIGHT
                if player == "black" \
                and piece == Piece.BLACK_PAWN \
                and row < 7 \
                and col < 7 \
                and self.get_piece(row + 1, col + 1).colour == "white":
                    rtn.append(self.with_pieces([
                        PiecePosition(row,col,Piece.AIR),
                        PiecePosition(row + 1, col + 1, Piece.BLACK_PAWN),
                    ]))

        return rtn

    def show(self) -> None:
        print(self.board.argmax(dim=2))

class Game:
    state: State

    def __init__(self) -> None:
        self.state = State.get_starting_state()

from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass, replace
from typing import List, Tuple
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

    def with_pieces(self, pieces: List[Tuple[int,int,Piece]]) -> State:
        new_board = self.board.clone()
        for piece in pieces:
            new_board[piece[0], piece[1], :] = 0
            new_board[piece[0], piece[1], piece[2].value] = 1
        return replace(self, board=new_board)

    def get_piece(self, row: int, col: int) -> Piece:
        return Piece(int(self.board[row, col].argmax()))

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row <= 7 and 0 <= col <= 7

    def get_valid_moves(self, player: PlayerColour) -> List[State]:
        rtn: List[State] = []
        candidate_pawn_moves: List[State] = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                def move(target_row: int, target_col: int) -> None:
                    rtn.append(self.with_pieces([
                        (row,col,Piece.AIR),
                        (target_row, target_col, piece)
                    ]))
                if piece.colour != player: continue
                if piece.role == "king":
                    for i in range(-1,2):
                        for j in range(-1, 2):
                            if not self.in_bounds(row+i,col+j): continue
                            target = self.get_piece(row+i, col+j)
                            if target.colour != player:
                                move(row+i, col+j)
                            
                # WHITE PAWN ADVANCE
                if piece == Piece.WHITE_PAWN \
                and row > 0 \
                and self.get_piece(row - 1, col) == Piece.AIR:
                    move(row-1, col)
                # WHITE PAWN CAPTURE LEFT
                if piece == Piece.WHITE_PAWN \
                and row > 0 \
                and col > 0 \
                and self.get_piece(row - 1, col - 1).colour == "black":
                    move(row-1, col-1)
                # WHITE PAWN CAPTURE RIGHT
                if piece == Piece.WHITE_PAWN \
                and row > 0 \
                and col < 7 \
                and self.get_piece(row - 1, col + 1).colour == "black":
                    move(row-1, col+1)
                # BLACK PAWN ADVANCE
                if piece == Piece.BLACK_PAWN \
                and row < 7 \
                and self.get_piece(row + 1, col) == Piece.AIR:
                    move(row+1,col)
                # BLACK PAWN CAPTURE LEFT
                if piece == Piece.BLACK_PAWN \
                and row < 7 \
                and col > 0 \
                and self.get_piece(row + 1, col - 1).colour == "white":
                    move(row+1,col-1)
                # BLACK PAWN CAPTURE RIGHT
                if piece == Piece.BLACK_PAWN \
                and row < 7 \
                and col < 7 \
                and self.get_piece(row + 1, col + 1).colour == "white":
                    move(row+1,col+1)

                # STRAIGHTS
                if piece.role == "queen" or piece.role == "rook":
                    # move left
                    for target_col in range(col-1,-1,-1):
                        target = self.get_piece(row, target_col)
                        if target.colour != player:
                            move(row,target_col)
                        if target != Piece.AIR: break
                    # move right
                    for target_col in range(col+1, 8):
                        target = self.get_piece(row, target_col)
                        if target.colour != player:
                            move(row, target_col)
                        if target != Piece.AIR: break
                    # move up
                    for target_row in range(row-1,-1,-1):
                        target = self.get_piece(target_row, col)
                        if target.colour != player:
                            move(target_row, col)
                        if target != Piece.AIR: break
                    # move down
                    for target_row in range(row+1,8):
                        target = self.get_piece(target_row, col)
                        if target.colour != player:
                            move(target_row, col)
                        if target != Piece.AIR: break           

                # DIAGS
                if piece.role == "queen" or piece.role == "bishop":
                    # up-right
                    for i in range(1,8):
                        if not self.in_bounds(row-i, col+i): continue
                        target = self.get_piece(row-i, col+i)
                        if target.colour != player:
                            move(row-i, col+i)
                        if target != Piece.AIR: break
                    # up-left
                    for i in range(1,8):
                        if not self.in_bounds(row-i, col-i): continue
                        target = self.get_piece(row-i, col-i)
                        if target.colour != player:
                            move(row-i, col-i)
                        if target != Piece.AIR: break
                    # down-right
                    for i in range(1,8):
                        if not self.in_bounds(row+i, col+i): continue
                        target = self.get_piece(row+i, col+i)
                        if target.colour != player:
                            move(row+i, col+i)
                        if target != Piece.AIR: break
                    # down-left
                    for i in range(1,8):
                        if not self.in_bounds(row+i, col-i): continue
                        target = self.get_piece(row+i, col-i)
                        if target.colour != player:
                            move(row+i, col-i)
                        if target != Piece.AIR: break
                            
                # KNIGHTS
                if piece.role == "knight":
                    for i,j in [
                        (-2,1),
                        (-1,2),
                        (1,2),
                        (2,1),
                        (2,-1),
                        (1,-2),
                        (-1,-2),
                        (-2,-1)
                    ]:
                        if not self.in_bounds(row+i, col+j): continue
                        target = self.get_piece(row+i, col-i)
                        if target.colour != player:
                            move(row+i, col+j)


        for move in candidate_pawn_moves:
            # WHITE PAWN PROMOTE
            if player == "white":
                for col in range(0,8):
                    piece = move.get_piece(0, col)
                    if piece == Piece.WHITE_PAWN:
                        for promoted in [
                            Piece.WHITE_BISHOP,
                            Piece.WHITE_KNIGHT,
                            Piece.WHITE_ROOK,
                            Piece.WHITE_QUEEN
                        ]:
                            # move only valid by replacing the pawn with the promoted piece
                            rtn.append(move.with_pieces([(0,col, promoted)]))
            
            # BLACK PAWN PROMOTE
            if player == "black":
                for col in range(0,8):
                    piece = move.get_piece(7, col)
                    if piece == Piece.BLACK_PAWN:
                        for promoted in [
                            Piece.BLACK_BISHOP,
                            Piece.BLACK_KNIGHT,
                            Piece.BLACK_ROOK,
                            Piece.BLACK_QUEEN
                        ]:
                            # move only valid by replacing the pawn with the promoted piece
                            rtn.append(move.with_pieces([(7,col, promoted)]))
        
        return rtn

    def show(self) -> None:
        print(self.board.argmax(dim=2))

class Game:
    state: State

    def __init__(self) -> None:
        self.state = State.get_starting_state()

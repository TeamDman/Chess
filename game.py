from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Tuple, Union
from typing_extensions import Literal
from torch import Tensor
import torch
from definitions import Piece, PlayerColour, Role

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
BoardTensorBatch = Tensor
BoardMetadataTensorBatch = Tensor
QValueTensorBatch = Tensor

@dataclass(frozen=True)
class StateTensorBatch:
    board: BoardTensorBatch
    metadata: BoardMetadataTensorBatch

    @property
    def batch_size(self) -> int:
        assert self.board.shape[0] == self.metadata.shape[0]
        return self.board.shape[0]

    @staticmethod
    def cat(batches: List[StateTensorBatch]) -> StateTensorBatch:
        return StateTensorBatch(
            board=torch.cat([s.board for s in batches]),
            metadata=torch.cat([s.metadata for s in batches]),
        )

    def to(self, device: torch.device) -> StateTensorBatch:
        return StateTensorBatch(
            board=self.board.to(device),
            metadata=self.metadata.to(device),
        )

    def zero_(self) -> StateTensorBatch:
        self.board.zero_()
        self.metadata.zero_()
        return self

    def repeat_batch(self, times: int) -> None:
        return StateTensorBatch(
            board=self.board.repeat((times,1,1,1)),
            metadata=self.metadata.repeat((times,1)),
        )
ActionTensorBatch = StateTensorBatch

BOARD_LEN = 8*8*len(Piece)
BOARD_SHAPE = (-1,8,8,len(Piece))
META_LEN = 7
META_SHAPE = (-1, META_LEN)

@dataclass(frozen=True)
class State:
    board: BoardTensor
    black_left_rook_moved: bool = False
    black_right_rook_moved: bool = False
    black_king_moved: bool = False
    white_left_rook_moved: bool = False
    white_right_rook_moved: bool = False
    white_king_moved: bool = False
    is_white_turn: bool = True

    @property
    def player(self) -> str:
        return "white" if self.is_white_turn else "black"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, State): return False
        if self.black_left_rook_moved != __o.black_left_rook_moved: return False
        if self.black_right_rook_moved != __o.black_right_rook_moved: return False
        if self.black_king_moved != __o.black_king_moved: return False
        if self.white_left_rook_moved != __o.white_left_rook_moved: return False
        if self.white_right_rook_moved != __o.white_right_rook_moved: return False
        if self.white_king_moved != __o.white_king_moved: return False
        if self.is_white_turn != __o.is_white_turn: return False
        return torch.equal(self.board, __o.board)

    def as_player(self, player: PlayerColour) -> State:
        return self if player == self.player else replace(self, is_white_turn = not self.is_white_turn)

    def as_other_player(self, player: PlayerColour) -> State:
        return replace(self, is_white_turn = player == "black")

    def as_tensor_batch(self) -> StateTensorBatch:
        return StateTensorBatch(
            board=self.board.unsqueeze(0),
            metadata=torch.as_tensor([(
                self.black_left_rook_moved,
                self.black_right_rook_moved,
                self.black_king_moved,
                self.white_left_rook_moved,
                self.white_right_rook_moved,
                self.white_king_moved,
                self.is_white_turn,
            )], dtype=torch.float32),
        )

    @staticmethod
    def from_tensors(board: BoardTensorBatch, meta: BoardMetadataTensorBatch) -> List[State]:
        assert board.shape[0] == meta.shape[0]
        batch_size = board.shape[0]
        rtn = []
        for i in range(batch_size):
            rtn.append(State(
                board=board[i],
                black_left_rook_moved=meta[i][0],
                black_right_rook_moved=meta[i][1],
                black_king_moved=meta[i][2],
                white_left_rook_moved=meta[i][3],
                white_right_rook_moved=meta[i][4],
                white_king_moved=meta[i][5],
                is_white_turn=meta[i][6],
            ))
        return rtn
            

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

    def get_valid_moves(self, enforce_check=True) -> List[State]:
        rtn: List[State] = []
        candidate_pawn_moves: List[State] = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece == Piece.AIR: continue
                def move(target_row: int, target_col: int, lst:List=rtn) -> None:
                    lst.append(self.with_pieces([
                        (row,col,Piece.AIR),
                        (target_row, target_col, piece)
                    ]))
                if piece.colour != self.player: continue
                if piece.role == "king":
                    for i in range(-1,2):
                        for j in range(-1, 2):
                            if not self.in_bounds(row+i,col+j): continue
                            target = self.get_piece(row+i, col+j)
                            if target.colour != self.player:
                                move(row+i, col+j)
                            
                # WHITE PAWN ADVANCE
                if piece == Piece.WHITE_PAWN \
                and row > 0 \
                and self.get_piece(row - 1, col) == Piece.AIR:
                    move(row-1, col, candidate_pawn_moves)
                # WHITE PAWN CAPTURE LEFT
                if piece == Piece.WHITE_PAWN \
                and row > 0 \
                and col > 0 \
                and self.get_piece(row - 1, col - 1).colour == "black":
                    move(row-1, col-1, candidate_pawn_moves)
                # WHITE PAWN CAPTURE RIGHT
                if piece == Piece.WHITE_PAWN \
                and row > 0 \
                and col < 7 \
                and self.get_piece(row - 1, col + 1).colour == "black":
                    move(row-1, col+1, candidate_pawn_moves)
                # BLACK PAWN ADVANCE
                if piece == Piece.BLACK_PAWN \
                and row < 7 \
                and self.get_piece(row + 1, col) == Piece.AIR:
                    move(row+1,col, candidate_pawn_moves)
                # BLACK PAWN CAPTURE LEFT
                if piece == Piece.BLACK_PAWN \
                and row < 7 \
                and col > 0 \
                and self.get_piece(row + 1, col - 1).colour == "white":
                    move(row+1,col-1, candidate_pawn_moves)
                # BLACK PAWN CAPTURE RIGHT
                if piece == Piece.BLACK_PAWN \
                and row < 7 \
                and col < 7 \
                and self.get_piece(row + 1, col + 1).colour == "white":
                    move(row+1,col+1, candidate_pawn_moves)

                # STRAIGHTS
                if piece.role == "queen" or piece.role == "rook":
                    # move left
                    for target_col in range(col-1,-1,-1):
                        target = self.get_piece(row, target_col)
                        if target.colour != self.player:
                            move(row,target_col)
                        if target != Piece.AIR: break
                    # move right
                    for target_col in range(col+1, 8):
                        target = self.get_piece(row, target_col)
                        if target.colour != self.player:
                            move(row, target_col)
                        if target != Piece.AIR: break
                    # move up
                    for target_row in range(row-1,-1,-1):
                        target = self.get_piece(target_row, col)
                        if target.colour != self.player:
                            move(target_row, col)
                        if target != Piece.AIR: break
                    # move down
                    for target_row in range(row+1,8):
                        target = self.get_piece(target_row, col)
                        if target.colour != self.player:
                            move(target_row, col)
                        if target != Piece.AIR: break           

                # DIAGS
                if piece.role == "queen" or piece.role == "bishop":
                    # up-right
                    for i in range(1,8):
                        if not self.in_bounds(row-i, col+i): continue
                        target = self.get_piece(row-i, col+i)
                        if target.colour != self.player:
                            move(row-i, col+i)
                        if target != Piece.AIR: break
                    # up-left
                    for i in range(1,8):
                        if not self.in_bounds(row-i, col-i): continue
                        target = self.get_piece(row-i, col-i)
                        if target.colour != self.player:
                            move(row-i, col-i)
                        if target != Piece.AIR: break
                    # down-right
                    for i in range(1,8):
                        if not self.in_bounds(row+i, col+i): continue
                        target = self.get_piece(row+i, col+i)
                        if target.colour != self.player:
                            move(row+i, col+i)
                        if target != Piece.AIR: break
                    # down-left
                    for i in range(1,8):
                        if not self.in_bounds(row+i, col-i): continue
                        target = self.get_piece(row+i, col-i)
                        if target.colour != self.player:
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
                        target = self.get_piece(row+i, col+j)
                        if target.colour != self.player:
                            move(row+i, col+j)


        for pstate in candidate_pawn_moves:
            # WHITE PAWN PROMOTE
            if self.player == "white":
                for col in range(0,8):
                    piece = pstate.get_piece(0, col)
                    if piece == Piece.WHITE_PAWN:
                        for promoted in [
                            Piece.WHITE_BISHOP,
                            Piece.WHITE_KNIGHT,
                            Piece.WHITE_ROOK,
                            Piece.WHITE_QUEEN
                        ]:
                            # move only valid by replacing the pawn with the promoted piece
                            rtn.append(pstate.with_pieces([(0,col, promoted)]))
            
            # BLACK PAWN PROMOTE
            if self.player == "black":
                for col in range(0,8):
                    piece = pstate.get_piece(7, col)
                    if piece == Piece.BLACK_PAWN:
                        for promoted in [
                            Piece.BLACK_BISHOP,
                            Piece.BLACK_KNIGHT,
                            Piece.BLACK_ROOK,
                            Piece.BLACK_QUEEN
                        ]:
                            # move only valid by replacing the pawn with the promoted piece
                            rtn.append(pstate.with_pieces([(7,col, promoted)]))
        
        # todo: castling
        # todo: en-passant, double-move first pawn move

        # flip between player each turn
        rtn = [replace(v, is_white_turn=not v.is_white_turn) for v in rtn]
        
        # you may not move into check
        if enforce_check:
            rtn = [move for move in rtn if not move.is_check(self.player)]
        return rtn

    def show(self, style:Union[Literal["text"],Literal["svg"]] = "svg", **kwargs) -> None:
        if style == "text":
            print(self.board.argmax(dim=2))
        else:
            from IPython.display import display
            from rendering import board as board_svg
            display(board_svg(self, **kwargs))

    def count_piece(self, piece: Piece) -> bool:
        return (self.board.argmax(dim=2) == piece.value).sum()

    def is_check(self, player: PlayerColour) -> bool:
        king = Piece.WHITE_KING if player == "white" else Piece.BLACK_KING
        moves = self.as_other_player(player).get_valid_moves(enforce_check=False)
        return any([move for move in moves if move.count_piece(king) == 0]) 

    def is_stalemate(self) -> bool:
        return len(self.get_valid_moves(enforce_check=True)) == 0 and not self.is_check(self.player)

    def is_checkmate(self) -> bool:
        return len(self.get_valid_moves(enforce_check=True)) == 0 and self.is_check(self.player)

    def capture_occurred(self, prev: State) -> bool:
        return bool(self.count_piece(Piece.AIR) != prev.count_piece(Piece.AIR))

    def mask_for(self, piece: Piece) -> Tensor:
        return self.board.argmax(dim=2) == piece.value

    def pawn_moved(self, prev: State) -> bool:
        pawn = self.other("pawn")
        old = prev.mask_for(pawn)
        current = self.mask_for(pawn)
        return not torch.equal(old, current) and old.sum() == current.sum()

    def sufficient_material(self) -> bool:
        if self.count_piece(Piece.WHITE_PAWN) > 0 or \
            self.count_piece(Piece.BLACK_PAWN) > 0 or \
            self.count_piece(Piece.WHITE_ROOK) > 0 or \
            self.count_piece(Piece.BLACK_ROOK) > 0 or \
            self.count_piece(Piece.WHITE_QUEEN) > 0 or \
            self.count_piece(Piece.BLACK_QUEEN) > 0:
                return True
        white_knight = self.count_piece(Piece.WHITE_KNIGHT)
        white_bishop = self.count_piece(Piece.WHITE_BISHOP)
        black_knight = self.count_piece(Piece.BLACK_KNIGHT)
        black_bishop = self.count_piece(Piece.BLACK_BISHOP)
        return not ((
            white_knight == 1 and white_bishop == 0 or \
            white_bishop == 1 and white_knight == 0 or \
            white_bishop == 0 and white_knight == 0
        ) and (
            black_knight == 1 and black_bishop == 0 or \
            black_bishop == 1 and black_knight == 0 or \
            black_bishop == 0 and black_knight == 0
        ))


    def mine(self, role: Role) -> Piece:
        return [piece for piece in Piece if piece.name == f"{self.player}_{role}".upper()][0]

    def other(self, role: Role) -> Piece:
        return self.as_other_player(self.player).mine(role)


Action = State

class Game:
    state: State
    def __init__(self) -> None:
        self.state = State.get_starting_state()

    def step(self) -> None:
        pass

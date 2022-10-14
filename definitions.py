from typing import List, Union
from typing_extensions import Literal
PlayerColour = Union[Literal["white"], Literal["black"]]
from enum import unique, auto, Enum

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

    @property
    def symbol(self) -> str:
        name = self.name[6:]
        letter = self.name[6:7]
        if name == "KNIGHT":
            letter = "N"
        colour = self.name[0:5]
        if colour == "BLACK":
            letter = letter.lower()
        return letter

    @property
    def colour(self) -> PlayerColour:
        return self.name[0:5].lower()


white_pieces = [
    Piece.WHITE_PAWN,
    Piece.WHITE_ROOK,
    Piece.WHITE_KNIGHT,
    Piece.WHITE_BISHOP,
    Piece.WHITE_QUEEN,
    Piece.WHITE_KING,
]

black_pieces = [
    Piece.BLACK_PAWN,
    Piece.BLACK_ROOK,
    Piece.BLACK_KNIGHT,
    Piece.BLACK_BISHOP,
    Piece.BLACK_QUEEN,
    Piece.BLACK_KING,
]
from __future__ import annotations
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
        letter = self.role[0].upper()
        if self.role == "knight":
            letter = "N"
        colour = self.colour
        if colour == "black":
            letter = letter.lower()
        return letter

    @property
    def role(self) -> Role:
        return self.name[6:].lower()

    @property
    def colour(self) -> Union[PlayerColour, None]:
        if self == Piece.AIR:
            return None
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

Role = Union[
    Literal["rook"],
    Literal["knight"],
    Literal["king"],
    Literal["queen"],
    Literal["bishop"],
    Literal["pawn"]
]
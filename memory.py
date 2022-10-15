from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Generic, List, TypeVar
from torch import Tensor
import torch

if TYPE_CHECKING:
    from game import BatchedBoardTensor, BatchedBoardMetadataTensor

@dataclass(frozen=True)
class TransitionTensorBatch:
    board: BatchedBoardTensor
    board_meta: BatchedBoardMetadataTensor
    next_board: BatchedBoardTensor
    next_board_meta: BatchedBoardMetadataTensor
    reward: Tensor # float
    terminal: Tensor # bool

T = TypeVar("T")
class ReplayMemory(ABC, Generic[T]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def push(self, v: T) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[T]:
        pass

    @abstractmethod
    def get_max_len(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class DequeReplayMemory(ReplayMemory, Generic[T]):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([],maxlen=capacity)

    def push(self, v: T) -> None:
        self.memory.append(v)

    def sample(self, batch_size: int) -> List[T]:
        return random.sample(self.memory, batch_size)

    def get_max_len(self) -> int:
        return self.memory.maxlen

    def __len__(self) -> int:
        return len(self.memory)
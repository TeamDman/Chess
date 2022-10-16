from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Generic, List, TypeVar
from torch import Tensor
import torch

from game import StateTensorBatch

if TYPE_CHECKING:
    from game import BoardTensorBatch, BoardMetadataTensorBatch

@dataclass(frozen=True)
class TransitionTensorBatch:
    state: StateTensorBatch
    action: StateTensorBatch
    next_state: StateTensorBatch
    reward: Tensor # float
    terminal: Tensor # bool

    @staticmethod
    def cat(batches: List[TransitionTensorBatch]) -> TransitionTensorBatch:
        return TransitionTensorBatch(
            state=StateTensorBatch.cat([v.state for v in batches]),
            action=StateTensorBatch.cat([v.action for v in batches]),
            next_state=StateTensorBatch.cat([v.next_state for v in batches]),
            reward=torch.cat([v.reward for v in batches]),
            terminal=torch.cat([v.terminal for v in batches]),
        )

    def to(self, device: torch.device) -> TransitionTensorBatch:
        return TransitionTensorBatch(
            state=self.state.to(device),
            action=self.action.to(device),
            next_state=self.next_state.to(device),
            reward=self.reward.to(device),
            terminal=self.terminal.to(device),
        )

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
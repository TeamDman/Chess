from dataclasses import dataclass
from agents import Agent
from definitions import Piece
from game import State

from memory import ReplayMemory, TransitionTensorBatch
from tqdm.notebook import tqdm

@dataclass
class TrainerConfig:
    batch_size: int
    white_memory: ReplayMemory[TransitionTensorBatch]
    black_memory: ReplayMemory[TransitionTensorBatch]
    warmup_steps: int
    optimization_steps: int
    reward_gamma: float
    
    soft_update_tau: float 
    epsilon_start: float 
    epsilon_end: float 
    epsilon_decay: float 

    white_agent: Agent
    black_agent: Agent

class Trainer:
    config: TrainerConfig
    state: State

    def __init__(self, config: TrainerConfig) -> None:
        assert config.white_memory.get_max_len() > config.batch_size
        assert config.black_memory.get_max_len() > config.batch_size
        self.config = config

    def prepare_next_episode(self) -> None:
        self.state = State.get_starting_state()

    def exploration_step(self) -> None:
        # white player's turn
        state0 = self.state
        state1 = self.config.white_agent.take_turn(self.state)
        board1, board_meta1 = state0.as_tensor_batch()
        terminal1 = state1.count_piece(Piece.BLACK_KING) == 0
        reward1 = 0 if not terminal1 else 1

        if terminal1:
            next_board, next_board_meta = State.get_empty_state().as_tensor_batch()
            # white wins, next state is None
            self.config.white_memory.push(TransitionTensorBatch(
                board=board1,
                board_meta=board_meta1,
                next_board=next_board.zero_(),
                next_board_meta=next_board_meta.zero_(),
                reward=reward1,
                terminal=terminal1,
            ))
            self.prepare_next_episode()
            return

        # black player's turn
        state2 = self.config.black_agent.take_turn(state1)
        board2, board_meta2 = state0.as_tensor_batch()
        terminal2 = state2.count_piece(Piece.WHITE_KING) == 0
        reward2 = -1 if terminal2 else 0
        if terminal2:
            board2.zero_()
            board_meta2.zero_()

        self.config.white_memory.push(TransitionTensorBatch(
            board=board1,
            board_meta=board_meta1,
            next_board=board2,
            next_board_meta=board_meta2,
            reward=reward2,
            terminal=terminal2
        ))

        self.state = state2
        
        

    def optimization_step(self) -> None:
        pass

    def train(self) -> None:
        self.prepare_next_episode()
        print("Warming up")
        warmup_steps = max(self.config.warmup_steps, self.config.batch_size)
        warmup_steps = warmup_steps - len(self.config.white_memory)
        warmup_steps = max(0, warmup_steps)
        for _ in tqdm(range(warmup_steps)):
            self.exploration_step()
        print("Warmup complete~!")

        for _ in tqdm(range(self.config.optimization_steps)):
            self.optimization_step()
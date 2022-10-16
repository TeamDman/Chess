import random
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import List, Union
from typing_extensions import Literal
from agents import Agent
from definitions import Piece
from game import Action, ActionTensorBatch, QValueTensorBatch, State, StateTensorBatch
import torch
from torch import Tensor
from memory import ReplayMemory, TransitionTensorBatch
from tqdm.notebook import tqdm
import math
from utils import get_device

criterion = torch.nn.MSELoss()


@dataclass
class TrainerConfig:
    batch_size: int
    memory: ReplayMemory[TransitionTensorBatch]
    warmup_steps: int
    optimization_steps: int
    reward_gamma: float
    
    soft_update_tau: float 
    epsilon_start: float 
    epsilon_end: float 
    epsilon_decay: float 

    starting_state: State

    update_policy_interval: int
    policy_update_type: Union[Literal["hard"], Literal["soft"]]

    agent: Agent

class Trainer:
    config: TrainerConfig
    state: State
    state_after_white: Union[State,None]
    black_action: Action
    episode: List[State]
    optim_step: int
    turns_since_last_capture_or_pawn_move: int

    def __init__(self, config: TrainerConfig) -> None:
        assert config.memory.get_max_len() > config.batch_size
        self.config = config
        self.optim_step = 0
        self.turns_since_last_capture_or_pawn_move = 0


    def get_epsilon_threshold(self, step) -> float:
        # https://www.desmos.com/calculator/kkt49vdqbj
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-1. * step / self.config.epsilon_decay)
 
    def plot_epsilon_threshold(self, num_steps:int) -> None:
        plt.title("epsilon threshold")
        points = [self.get_epsilon_threshold(i) for i in range(num_steps)]
        plt.yticks(torch.arange(0,10,step=0.1))
        plt.plot(points)
        plt.show()

    def get_action(self, agent: Agent, state: State) -> Action:
        if random.random() > self.get_epsilon_threshold(self.optim_step):
            return agent.get_action(state)
        else:
            return agent.get_random_action(state)
            
    def prepare_next_episode(self) -> None:
        self.state = self.config.starting_state
        self.state_after_white = None
        self.episode = [self.state]

    def exploration_step(self) -> None:
        state_before_white = self.state
        previous_state_before_black = self.state_after_white

        # white take turn
        assert state_before_white.is_white_turn
        white_action = self.get_action(self.config.agent, state_before_white)
        state_after_white = self.config.agent.take_action(state_before_white, white_action)
        assert not state_after_white.is_white_turn
        self.episode.append(state_after_white)

        # 50-move-rule
        if state_after_white.pawn_moved(state_before_white) \
        or state_after_white.capture_occurred(state_before_white):
            self.turns_since_last_capture_or_pawn_move += 1
        else:
            self.turns_since_last_capture_or_pawn_move = 0

        # check if game should end
        white_win = state_after_white.is_checkmate()
        repetition = len([1 for v in self.episode if v == state_after_white]) >= 3
        fiftymove = self.turns_since_last_capture_or_pawn_move == 50
        insufficient = not state_after_white.sufficient_material()
        stalemate = state_after_white.is_stalemate()
        if white_win or repetition or fiftymove or insufficient or stalemate:
            # track white's action as terminal in the replay
            self.config.memory.push(TransitionTensorBatch(
                state=state_before_white.as_tensor_batch(),
                action=white_action.as_tensor_batch(),
                next_state=State.get_empty_state().as_tensor_batch().zero_(),
                reward=torch.as_tensor(1 if white_win else 0, dtype=torch.float32).unsqueeze(0),
                terminal=torch.as_tensor(True, dtype=torch.bool).unsqueeze(0),
            ))
            if previous_state_before_black is not None:
                # track black's action as terminal in the replay
                self.config.memory.push(TransitionTensorBatch(
                    state=previous_state_before_black.as_tensor_batch(),
                    action=self.black_action.as_tensor_batch(),
                    next_state=State.get_empty_state().as_tensor_batch().zero_(),
                    reward=torch.as_tensor(-1 if white_win else 0, dtype=torch.float32).unsqueeze(0),
                    terminal=torch.as_tensor(True, dtype=torch.bool).unsqueeze(0),
                ))
            self.prepare_next_episode()
            return

        if previous_state_before_black is not None:
            # track black's action (from the previous step) in the replay
            self.config.memory.push(TransitionTensorBatch(
                state=previous_state_before_black.as_tensor_batch(),
                action=self.black_action.as_tensor_batch(),
                next_state=state_after_white.as_tensor_batch(),
                reward=torch.as_tensor(0, dtype=torch.float32).unsqueeze(0),
                terminal=torch.as_tensor(False, dtype=torch.bool).unsqueeze(0),
            ))

        # black take turn
        assert not state_after_white.is_white_turn
        self.black_action = self.get_action(self.config.agent, state_after_white)
        state_after_black = self.config.agent.take_action(state_after_white, self.black_action)
        assert state_after_black.is_white_turn
        self.episode.append(state_after_black)
        
        # 50-move rule
        if state_after_black.pawn_moved(state_after_white) \
        or state_after_black.capture_occurred(state_after_white):
            self.turns_since_last_capture_or_pawn_move += 1
        else:
            self.turns_since_last_capture_or_pawn_move = 0

        # check if game should end
        black_win = state_after_black.count_piece(Piece.WHITE_KING) == 0
        repetition = len([1 for v in self.episode if v == state_after_black]) >= 3
        fiftymove = self.turns_since_last_capture_or_pawn_move == 50
        insufficient = not state_after_black.sufficient_material()
        stalemate = state_after_black.is_stalemate()
        if black_win or repetition or fiftymove or insufficient or stalemate:
            # track white's action as terminal in the replay
            self.config.memory.push(TransitionTensorBatch(
                state=state_before_white.as_tensor_batch(),
                action=white_action.as_tensor_batch(),
                next_state=State.get_empty_state().as_tensor_batch().zero_(),
                reward=torch.as_tensor(-1 if black_win else 0, dtype=torch.float32).unsqueeze(0),
                terminal=torch.as_tensor(True, dtype=torch.bool).unsqueeze(0)
            ))
            # track black's action as terminal in the replay
            self.config.memory.push(TransitionTensorBatch(
                state=previous_state_before_black.as_tensor_batch(),
                action=self.black_action.as_tensor_batch(),
                next_state=State.get_empty_state().as_tensor_batch().zero_(),
                reward=torch.as_tensor(1 if white_win else 0, dtype=torch.float32).unsqueeze(0),
                terminal=torch.as_tensor(True, dtype=torch.bool).unsqueeze(0),
            ))
            self.prepare_next_episode()
            return

        # track white's action in the replay
        self.config.memory.push(TransitionTensorBatch(
            state=state_before_white.as_tensor_batch(),
            action=white_action.as_tensor_batch(),
            next_state=state_after_black.as_tensor_batch(),
            reward=torch.as_tensor(0, dtype=torch.float32).unsqueeze(0),
            terminal=torch.as_tensor(False, dtype=torch.bool).unsqueeze(0)
        ))




        self.state = state_after_black
        self.state_after_white = state_after_white

        

    def optimization_step(self) -> None:
        batch = self.config.memory.sample(self.config.batch_size)
        batch = TransitionTensorBatch.cat(batch).to(get_device())

        with torch.no_grad():
            proto_actions: ActionTensorBatch = self.config.agent.sticky_actor(batch.next_state)
            proto_q_values: QValueTensorBatch = self.config.agent.sticky_critic(batch.next_state, proto_actions)
            not_terminal = batch.terminal == False
            proto_q_values[not_terminal] += self.config.reward_gamma * proto_q_values[not_terminal]
        
        self.config.agent.critic.zero_grad()
        batch_q_values = self.config.agent.critic(batch.state, batch.action)
        value_loss: Tensor = criterion(batch_q_values, proto_q_values)
        print(f"value_loss={value_loss:.4f} ", end="")
        value_loss.backward()
        self.config.agent.critic_optimizer.step()

        self.config.agent.actor.zero_grad()
        policy_loss: Tensor = -1 * self.config.agent.critic(
            batch.state,
            self.config.agent.actor(batch.state)
        )
        policy_loss = policy_loss.mean()
        print(f"policy_loss={policy_loss:.4f} ", end="")
        policy_loss.backward()
        self.config.agent.actor_optimizer.step()        


        # Soft update wasn't training fast, trying hard update
        should_update_policy = self.optim_step % self.config.update_policy_interval == 0
        if should_update_policy:
            if self.config.policy_update_type == "soft":
                # from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py#L26
                def soft_update(target, source, tau):
                    for target_param, param in zip(target.parameters(), source.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - tau) + param.data * tau
                        )
                soft_update(self.config.agent.sticky_actor, self.config.agent.actor, self.config.soft_update_tau)
                soft_update(self.config.agent.sticky_critic, self.config.agent.critic, self.config.soft_update_tau)
            elif self.config.policy_update_type == "hard":
                self.config.agent.sticky_actor.load_state_dict(self.config.agent.actor.state_dict())
                self.config.agent.sticky_critic.load_state_dict(self.config.agent.critic.state_dict())
            else:
                raise ValueError(f"unknown policy update type: \"{self.config.policy_update_type}\"")
        print()
        self.optim_step += 1

    def train(self) -> None:
        self.prepare_next_episode()
        print("Warming up")
        warmup_steps = max(self.config.warmup_steps, self.config.batch_size)
        warmup_steps = warmup_steps - len(self.config.memory)
        warmup_steps = max(0, warmup_steps)
        for _ in tqdm(range(warmup_steps)):
            self.exploration_step()
        print("Warmup complete~!")
        print("Beginning optimization steps")
        self.optim_step = 0
        for _ in tqdm(range(self.config.optimization_steps)):
            self.optimization_step()
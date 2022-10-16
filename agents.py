from dataclasses import dataclass, replace
import random
from typing import List
import torch
from models import Actor, Critic
from utils import get_device
from game import Action, BoardMetadataTensorBatch, BoardTensorBatch, QValueTensorBatch, State, PlayerColour, StateTensorBatch

@dataclass
class AgentConfig:
    learning_rate: float
    num_proposals: int

class Agent:
    config: AgentConfig
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.actor = Actor().to(get_device())
        self.sticky_actor = Actor().to(get_device())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        
        self.critic = Critic().to(get_device())
        self.sticky_critic = Critic().to(get_device())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        self.sticky_actor.load_state_dict(self.actor.state_dict())
        self.sticky_critic.load_state_dict(self.critic.state_dict())
        

    @torch.no_grad()
    def get_next_state(self, current_state: State) -> State:
        state_batch = current_state.as_tensor_batch()
        assert state_batch.batch_size == 1
        # get proto actions from actor
        proto_action_batch: StateTensorBatch = self.actor(state_batch)
        assert proto_action_batch.batch_size == 1
        # find nearest real actions given the proto actions
        actions = self.collapse_proto_actions(current_state, proto_action_batch)
        assert len(actions) == 1 # batch of 1
        assert len(actions[0]) > 0 # must have legal moves
        actions = actions[0]
        candidate_action_batch = StateTensorBatch.cat([a.as_tensor_batch() for a in actions])
        # evaluate the real actions
        action_q_values: QValueTensorBatch = self.critic(
            state_batch.repeat_batch(candidate_action_batch.batch_size),
            candidate_action_batch
        )
        # find the best action
        best = action_q_values.argmax().cpu()
        return actions[best]
        # target_board = actions[best].board.cpu()
        # # find the action that corresponds to arriving at the best board
        # for potential in current_state.get_valid_moves():
        #     if torch.equal(potential.board, target_board):
        #         return potential
        # # if action not found, something went wrong
        # raise ValueError("collapsed proto action contained invalid move")

    def collapse_proto_actions(
        self,
        current_state: State,
        proto_actions: StateTensorBatch
    ) -> List[List[State]]:
        batch_size = proto_actions.batch_size
        rtn = []
        # get all legal moves
        candidates: List[State] = current_state.get_valid_moves()
        candidate_boards = torch.cat([s.board.unsqueeze(0) for s in candidates])
        # find the legal move with board closest to the proto move
        for i in range(batch_size):
            action_boards = proto_actions.board[i].unsqueeze(0).repeat(len(candidates),1,1,1)
            action_board_scores = (candidate_boards - action_boards).flatten(start_dim=1).sum(dim=1).abs()
            # sort with ascending difference
            vals, idxs = action_board_scores.sort()
            vals = vals.flatten().tolist()
            idxs = idxs.flatten().tolist()
            # take the legal actions closest to the proto-actions
            take = min(self.config.num_proposals, len(candidates))
            rtn.append([candidates[idxs[i]] for i in range(take)])
        return rtn

    def get_action(self, state: State) -> Action:
        # "action" is just the board state that the agent desires next
        return self.get_next_state(state)

    def take_action(self, state: State, action: Action) -> State:
        return action

    def get_random_action(self, state: State) -> Action:
        moves = state.get_valid_moves()
        assert len(moves) > 0
        return random.choice(moves)

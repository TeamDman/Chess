from dataclasses import dataclass, replace
from typing import List
import torch
from models import Actor, Critic
from utils import get_device
from game import BatchedBoardMetadataTensor, BatchedBoardTensor, BatchedQValueTensor, State, PlayerColour

@dataclass
class AgentConfig:
    learning_rate: float
    colour: PlayerColour
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
        batch = current_state.as_tensor_batch()
        # get proto actions from actor
        proto_action_batch: BatchedBoardTensor = self.actor(*batch)
        assert proto_action_batch[0].shape[0] == 1 # batch of 1
        assert proto_action_batch[1].shape[0] == 1 # batch of 1
        # find nearest real actions given the proto actions
        actions = self.collapse_proto_actions(current_state, *proto_action_batch)
        assert len(actions) == 1 # batch of 1
        action_batch = State.cat(actions[0])
        # evaluate the real actions
        action_q_values: BatchedQValueTensor = self.critic(*batch, *action_batch)
        # find the best board
        best = action_q_values.argmax().cpu()
        target_board = actions[0][best].board.cpu()
        # find the action that corresponds to arriving at the best board
        for potential in current_state.get_valid_moves(self.config.colour):
            if torch.equal(potential.board, target_board):
                return potential
        # if action not found, something went wrong
        raise ValueError("collapsed proto action contained invalid move")

    def collapse_proto_actions(
        self,
        current_state: State,
        proto_boards: BatchedBoardTensor,
        proto_metas: BatchedBoardMetadataTensor,
    ) -> List[List[State]]:
        assert proto_boards.shape[0] == proto_metas.shape[0]
        batch_size = proto_boards.shape[0]
        rtn = []
        # get all legal moves
        candidates: List[State] = current_state.get_valid_moves(self.config.colour)
        # get the proto moves as board tensors
        candidate_boards = torch.cat([s.board.unsqueeze(0) for s in candidates])
        for i in range(batch_size):
            # find the legal move with board closest to the proto move
            proto_cat = proto_boards[i].unsqueeze(0).repeat(len(candidates),1,1,1)
            scores = (candidate_boards - proto_cat).flatten(start_dim=1).sum(dim=1).abs()
            # sort with ascending difference
            vals, idxs = scores.sort()
            vals = vals.flatten().tolist()
            idxs = idxs.flatten().tolist()
            # take according to least-different-from-proto legal moves
            take = min(self.config.num_proposals, len(candidates))
            rtn.append([candidates[idxs[i]] for i in range(take)])
        return rtn

    def take_turn(self, state: State) -> State:
        return self.get_next_state(state)

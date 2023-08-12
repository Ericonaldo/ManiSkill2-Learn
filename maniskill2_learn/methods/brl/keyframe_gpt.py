"""
Code for the model architecture of KeyframeGPT, based on the CoTPC implementation, where we replace the history input as (s-a) histories.
Some of the key hyper-parameters are explained in GPTConfig.

References:
(1) https://github.com/karpathy/minGPT
(2) https://github.com/kzl/decision-transformer
(3) https://github.com/SeanJia/CoTPC
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from maniskill2_learn.networks.modules.block_utils import SimpleMLP as MLP
import numpy as np


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """

    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    def __init__(self, **kwargs):

        assert kwargs['model_type'] in ['s', 's+a'], \
            f"Unsupported model_type: {kwargs['model_type']}" # Determine the history input type
        
        self.block_size = kwargs["block_size"]*2 if "+a" in kwargs['model_type'] else kwargs["block_size"]
        self.len_history = kwargs["hist_horizon"]
        self.model_type = kwargs['model_type']

        if "+a" in self.model_type:
            self.len_history += (kwargs["hist_horizon"]-1)

        kwargs.pop("block_size")
        kwargs.pop("hist_horizon")
        kwargs.pop("model_type")

        # Set up other attributes.
        for k,v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttentionWithHist(nn.Module):
    """
    A multi-head masked self-attention layer equipped with history query tokens for
    chain-of-thought predictive control. It is adapted from the minGPT repo.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        block_size = config.block_size + config.len_history
        self.register_buffer("mask", 
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        
        self.n_head = config.n_head
        self.model_type = config.model_type
        self.len_history = config.len_history

        # For the history query tokens, they are actually all-to-all, meaning
        # they can access to all future tokens during inference. 
        self.mask[:,:,:self.len_history] = 0.0

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))  # Masked attention

        att = F.softmax(att, dim=-1)
        att = torch.where(torch.isnan(att), torch.full_like(att, 0), att) # replace nan with 0.
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    A Transformer block with masks specified for the history query tokens.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionWithHist(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BlocksWithHist(nn.Module):
    """
    A wrapper class for a sequence of Transformer blocks with masks specified for 
    the learnable history query tokens.
    """

    def __init__(self, config):
        super().__init__()
        # Register all the individual blocks.
        self.block_list = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.model_type = config.model_type
        self.n_head = config.n_head
        self.len_history = config.len_history

    def forward(self, x):
        B, T, _ = x.shape
        
        output = []  # Also keep the intermediate results.
        for block in self.block_list:
            x = block(x)
            output.append(x)
        
        return x, output


class KeyframeGPTWithHist(nn.Module):
    """ 
    GPT implementation with the support of the learnable history query tokens,
    which is used for the chain-of-thought predictive control. Here, the context size
    is specified as block_size, which does not count the history query tokens. 
    """

    def __init__(self, config, state_dim=-1, action_dim=-1, use_first_state=False, pose_only=False, pose_dim=7):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        
        self.config = GPTConfig(**config)
        self.optim_cfg = self.config.optim_cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = self.config.model_type
        self.block_size = self.config.block_size
        self.len_history = self.config.len_history
        self.pose_only = pose_only
        self.use_first_state = use_first_state

        # Set up learnable position embedding synchronized for s and a tokens, as proposed
        # in Decision Transformer. We use a similar global+local position embedding design.
        p_size = self.config.block_size // 2 if '+a' in self.model_type else self.config.block_size
        self.local_pos_emb = nn.Parameter(torch.zeros(1, p_size, self.config.n_embd))
        if use_first_state:
            self.first_local_pos_emb = nn.Parameter(torch.zeros(1, 1, self.config.n_embd))
        self.global_pos_emb = nn.Parameter(
            torch.zeros(1, self.config.max_timestep, self.config.n_embd))

        self.drop = nn.Dropout(self.config.embd_pdrop)

        self.history_pos_emb = nn.Parameter(
            torch.zeros(1, self.len_history, self.config.n_embd))

        # Transformer (attention layers) with CoT.
        self.blocks = BlocksWithHist(self.config)
        
        # State embeddings.
        if self.state_dim > 1000:
            self.state_encoder = MLP(self.state_dim, self.config.n_embd, hidden_dims=[512, 256])
        else:
            self.state_encoder = MLP(self.state_dim, self.config.n_embd, hidden_dims=[256])
        
        # Action embeddings.
        if '+a' in self.model_type:
            self.action_encoder = MLP(self.action_dim, self.config.n_embd, hidden_dims=[256])

        # Keyframe Action predictor.
        self.ln = nn.LayerNorm(self.config.n_embd)
        self.pose_dim = pose_dim
        if self.pose_only:
            self.key_frame_state_predictor = MLP(self.config.n_embd, self.pose_dim+1, hidden_dims=[256,256]) # We only predict pose
        else:
            self.key_frame_action_predictor = MLP(self.config.n_embd, action_dim+1, hidden_dims=[256,256]) # Action + timestep diffrence
            self.key_frame_state_predictor = MLP(self.config.n_embd, state_dim, hidden_dims=[256,256])

        self.apply(self._init_weights)
        print(f"Total # of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Given state (and action) history, predict actions (and historys).
    # `timesteps` is used for the global+local position embedding design similar
    # to the one in Decision Transformer. `history_mask` is used so that the 
    # (all-to-all) history query tokens can attend to later tokens. 
    def forward(self, states, timesteps, actions=None, first_state=None): # Time steps should be the same shape as states
        B, T = states.shape[0], states.shape[1] # History length
        state_embeddings = self.state_encoder(states)
        if first_state is not None:
            first_state_embedding = self.state_encoder(first_state)

        # Embeddings for state (action, and history query) tokens.
        token_embeddings = torch.zeros([B, self.block_size, self.config.n_embd], 
                                       dtype=torch.float32, device=states.device)
        
        # If using action history as inputs: during training, all actions are
        # specified; during inference, only actions in the past are specified.
        # That is, the first action prediction has no action history as inputs. 
        if '+a' in self.model_type:
            if first_state is not None:
                token_embeddings[:,0:1,:] = first_state_embedding
                token_embeddings[:,1:T*2+1:2,:] = state_embeddings
            else:
                token_embeddings[:,:T*2:2,:] = state_embeddings
            if actions is not None: 
                # Assume the last action is not used as inputs during training.
                action_embeddings = self.action_encoder(actions[:,:T-1])
                if first_state is not None:
                    token_embeddings[:,2:T*2:2,:] = action_embeddings
                else:
                    token_embeddings[:,1:T*2-1:2,:] = action_embeddings
                    
        else:
            token_embeddings[:,:T,:] = state_embeddings

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps = timesteps[:,0] # Take the start timestep
        timesteps_rp = torch.repeat_interleave(timesteps[:,None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(
            global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb, 2, dim=1) \
            if '+a' in self.model_type else self.local_pos_emb
        if first_state is not None:
            local_pos_emb = torch.cat([self.first_local_pos_emb, local_pos_emb], dim=1)[:,:self.block_size]

        x = token_embeddings + global_pos_emb + local_pos_emb
        
        x = self.drop(x)
        x, intermediate_feats = self.blocks(x)
        key_state_act_preds = self.ln(x)
        #### key_act_preds = self.key_frame_action_predictor(x)
        #### if '+a' in self.model_type:
        ####     # Remove the extra tokens when in eval mode.
        ####     key_state_act_preds = key_state_act_preds[:,T*2-1:] # The next tokens after s+a history, should be s,a,s,a,s,...
        #### else:
        ####     key_state_act_preds = key_state_act_preds[:,T:] 
        
        # Remove the extra tokens (history) when in eval mode.
        if self.pose_only:
            # Simplify the model, we only predict the tcp pose
            key_state_preds = self.key_frame_state_predictor(key_state_act_preds[:,T:]) # The next tokens after s history
            key_act_preds = None
        else:
            if '+a' in self.model_type:  # The next tokens after s+a history should be s,a,s,a,s,...
                key_state_preds = self.key_frame_state_predictor(key_state_act_preds[:,T*2-1:self.block_size:2]) # The next tokens after s+a history
                key_act_preds = self.key_frame_action_predictor(key_state_act_preds[:,T*2:self.block_size:2])
            else: # The next tokens after s history, should be s,a,s,a,s,..
                key_state_preds = self.key_frame_state_predictor(key_state_act_preds[:,T:self.block_size:2]) # The next tokens after s history
                key_act_preds = self.key_frame_action_predictor(key_state_act_preds[:,T+1:self.block_size:2])
    
        return key_state_preds, key_act_preds, {}  # Action + timestep diffrence

    def configure_adamw_optimizers(self, extra_model=None):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                    
        extra_para_dict = {}
        if extra_model is not None:
            extra_para_dict.update({pn: p for pn, p in extra_model.named_parameters()})
            for mn, m in extra_model.named_modules():
                for pn, _ in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('local_pos_emb')
        no_decay.add('global_pos_emb')
        no_decay.add('history_pos_emb')
        if self.use_first_state:
            no_decay.add('first_local_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict.update(extra_para_dict)

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
                % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": self.optim_cfg['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=self.optim_cfg['init_lr'], 
            betas=(self.optim_cfg['beta1'], self.optim_cfg['beta2'])
        )
        return optimizer
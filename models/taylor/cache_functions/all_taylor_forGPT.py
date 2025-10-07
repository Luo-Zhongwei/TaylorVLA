# Besides, re-arrange the attention module
from torch.jit import Final
from timm.layers import use_fused_attn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        #self.fused_attn = False
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cache_dic, current, fresh_indices=None) -> torch.Tensor:
    # 0.4ms extra cost on A800, mainly tensor operations
        """
        fresh_indices: (B, fresh_ratio*N), the index tensor for the fresh tokens
        """

        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   #q: (B, num_heads, N, head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn= attn.softmax(dim=-1) 
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) 
        
        flops = (
            B * N * C * 3 * C * 2 # QKV projection
            + B * self.num_heads * N * self.head_dim  # Scale q
            + B * self.num_heads * N * N * self.head_dim * 2 # Q @ K
            + B * self.num_heads * N * N * 5 # Softmax
            + B * self.num_heads * N * N * self.head_dim * 2 # Attn @ V
            + B * N * C * C * 2 # Projection
        )
        cache_dic['flops']+=flops
        return x


from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
#from .token_merge import token_merge
import torch
def cache_cutfresh(cache_dic, tokens, current):
    '''
    Cut fresh tokens from the input tokens and update the cache counter.
    
    cache_dic: dict, the cache dictionary containing cache(main extra memory cost), indices and some other information.
    tokens: torch.Tensor, the input tokens to be cut.
    current: dict, the current step, layer, and module information. Particularly convenient for debugging.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)
    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio), 0.0, 1.0)
    # Generate the index tensor for fresh tokens
    score = score_evaluate(cache_dic, tokens, current)
    score = local_selection_with_bonus(score, 0.8, 2) # Uniform Spatial Distribution s4 mentioned in toca
    # 0.6, 2
    indices = score.argsort(dim=-1, descending=True)
    topk = int(fresh_ratio * score.shape[1])
    fresh_indices = indices[:, :topk]
    #stale_indices = indices[:, topk:]
    # (B, fresh_ratio *N)

    # Updating the Cache Frequency Score s3 mentioned in toca
    # stale tokens index + 1, fresh tokens index = 0
    cache_dic['cache_index'][-1][layer][module] += 1
    cache_dic['cache_index'][-1][layer][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    
    # not used in the final version
    cache_dic['cache_index']['layer_index'][module] += 1
    cache_dic['cache_index']['layer_index'][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    # select the fresh tokens out
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])

    if module in ['mlp', 'attn']:
        # cut out the fresh tokens
        fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)

        return fresh_indices, fresh_tokens
    
    else:
        # no need for this branch hhh.
        raise ValueError("Unrecognized module?", module)
    
def local_selection_with_bonus(score, bonus_ratio, grid_size=2):
    '''
    Uniform Spatial Distribution s4 mentioned in toca
    '''
    batch_size, num_tokens = score.shape
    image_size = int(num_tokens ** 0.5)
    block_size = grid_size * grid_size
    
    assert num_tokens % block_size == 0, "The number of tokens must be divisible by the block size."
    
    # Step 1: Reshape score to group it by blocks
    score_reshaped = score.view(batch_size, image_size // grid_size, grid_size, image_size // grid_size, grid_size)
    score_reshaped = score_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    score_reshaped = score_reshaped.view(batch_size, -1, block_size)  # [batch_size, num_blocks, block_size]
    
    # Step 2: Find the max token in each block
    max_scores, max_indices = score_reshaped.max(dim=-1, keepdim=True)  # [batch_size, num_blocks, 1]
    
    # Step 3: Create a mask to identify max score tokens
    mask = torch.zeros_like(score_reshaped)
    mask.scatter_(-1, max_indices, 1)  # Set mask to 1 at the max indices
    
    # Step 4: Apply the bonus only to the max score tokens
    score_reshaped = score_reshaped + (mask * max_scores * bonus_ratio)  # Apply bonus only to max tokens
    
    # Step 5: Reshape the score back to its original shape
    score_modified = score_reshaped.view(batch_size, image_size // grid_size, image_size // grid_size, grid_size, grid_size)
    score_modified = score_modified.permute(0, 1, 3, 2, 4).contiguous()
    score_modified = score_modified.view(batch_size, num_tokens)
    
    return score_modified


def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    for j in range(28):
        cache[-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}

    cache_dic['cache']                = cache
    cache_dic['flops']                = 0.0
    cache_dic['interval']             = model_kwargs['interval']
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['first_enhance']        = 2
    cache_dic['cache_counter']        = 0
    
    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [49]
    return cache_dic, current


def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    last_steps = (current['step'] <=2)
    first_steps = (current['step'] > (current['num_steps'] - cache_dic['first_enhance'] - 1))

    fresh_interval = cache_dic['interval']

    if (first_steps) or (cache_dic['cache_counter'] == fresh_interval - 1 ):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        #current['activated_times'].append(current['t'])
    
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Taylor'
        
import torch
from .force_scheduler import force_scheduler
def force_init(cache_dic, current, tokens):
    '''
    Initialization for Force Activation step.
    '''
    # reset the cache index to 0
    cache_dic['cache_index'][-1][current['layer']][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)
    force_scheduler(cache_dic, current)
    if current['layer'] == 0:
        cache_dic['cache_index']['layer_index'][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)
        
import torch
def force_scheduler(cache_dic, current):
    '''
    Force Activation Cycle Scheduler
    '''
    cache_dic['cal_threshold'] = cache_dic['interval']

        
import torch
def fresh_ratio_scheduler(cache_dic, current):
    '''
    Return the fresh ratio for the current step.
    '''
    fresh_ratio = cache_dic['fresh_ratio']
    fresh_ratio_schedule = cache_dic['fresh_ratio_schedule']
    step = current['step']
    num_steps = current['num_steps']
    threshold = cache_dic['interval']
    weight = 0.9
    if fresh_ratio_schedule == 'constant':
        return fresh_ratio
    elif fresh_ratio_schedule == 'linear':
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps)
    elif fresh_ratio_schedule == 'exp':
        #return 0.5 * (0.052 ** (step/num_steps))
        return fresh_ratio * (weight ** (step / num_steps))
    elif fresh_ratio_schedule == 'linear-mode':
        mode = (step % threshold)/threshold - 0.5
        mode_weight = 0.1
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps + mode_weight * mode)
    elif fresh_ratio_schedule == 'layerwise':
        return fresh_ratio * (1 + weight - 2 * weight * current['layer'] / 27)
    elif fresh_ratio_schedule == 'linear-layerwise':
        step_weight = 0.4 
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps

        layer_weight = 0.8
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 27

        module_weight = 2.5
        module_time_weight = 0.6
        module_factor = (1 - (1-module_time_weight) * module_weight) if current['module']=='attn' else (1 + module_time_weight * module_weight)
        
        return fresh_ratio * layer_factor * step_factor * module_factor
    
    elif fresh_ratio_schedule == 'ToCa':
        # Proposed scheduling method in toca.

        # step wise scheduling, we find there is little differece if change the weight of step factor, so this is not a key factor. 
        step_weight = 2.0 #0.4 #0.0 # 2.0
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps

        # layer wise scheduling, important. Meaning caculate more in the front layers, less in the back layers.
        layer_weight = -0.2#0.8 #0.0 # -0.2
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 27

        # Module wise scheduling, important. Meaning caculate more in the mlp module, less in the attn module.
        module_weight = 2.5 # no calculations for attn module (2.5 * 0.4 = 1.0), compuation is transformed to mlp module.
        module_time_weight = 0.6 # estimated from the time and flops of mlp and attn module, may change in different situations.
        module_factor = (1 - (1-module_time_weight) * module_weight) if current['module']=='attn' else (1 + module_time_weight * module_weight)
        
        return fresh_ratio * layer_factor * step_factor * module_factor

    else:
        raise ValueError("unrecognized fresh ratio schedule", fresh_ratio_schedule)

from .force_scheduler import force_scheduler
def global_force_fresh(cache_dic, current):
    '''
    Return whether to force fresh tokens globally.
    '''
    last_steps = (current['step'] <= 2)
    first_step = (current['step'] == (current['num_steps'] - 1))
    force_fresh = cache_dic['force_fresh']
    if not first_step:
        fresh_threshold = cache_dic['cal_threshold']
    else:
        fresh_threshold = cache_dic['interval']

    if force_fresh == 'global':
    # global force fresh means force activate all tokens in this step.
        return (first_step or (current['step']% fresh_threshold == 0))
    
    elif force_fresh == 'local':
    # fresh locally cause much worse results, for the misalignment of cache and computed tokens.
        return first_step
    elif force_fresh == 'none':
        return first_step
    else:
        raise ValueError("unrecognized force fresh strategy", force_fresh)
    

import torch
import torch.nn as nn
from .scores import attn_score, similarity_score, norm_score, kv_norm_score
def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    '''
    Return the score tensor (B, N) for the given tokens. Mainly include s1, (s2,) s3 mentioned in toca.
    '''

    #if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')):
    ## abandoned branch, if you want to explore the local force fresh strategy, this may help.
    #    force_fresh_mask = torch.as_tensor((cache_dic['cache_index'][-1][current['layer']][current['module']] >= 2 * cache_dic['interval']), dtype = int) # 2 because the threshold is for step, not module
    #    force_len = force_fresh_mask.sum(dim=1)
    #    force_indices = force_fresh_mask.argsort(dim = -1, descending = True)[:, :force_len.min()]
    #
    #    force_indices = force_indices[:, torch.randperm(force_indices.shape[1])]

    if cache_dic['cache_type'] == 'random':
        # select tokens randomly, but remember to keep the same for cfg and no cfg.
        score = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1], device=tokens.device)
        score = torch.cat([score, score], dim=0).to(tokens.device)

    elif cache_dic['cache_type'] == 'straight':
        # abandon the cache, just return 1 hhh, obviously no use.
        score = torch.ones(tokens.shape[0], tokens.shape[1]).to(tokens.device)
    
    elif cache_dic['cache_type'] == 'attention':
        # Recommended selection method in toca.

        # cache_dic['attn_map'][step][layer] (B, N, N), the last dimention has get softmaxed

        # calculate the attention score, for DiT, there is no cross-attention, so just self-attention score s1 applied.
        score = attn_score(cache_dic, current)

        # if you'd like to add some randomness to the score as SiTo does to avoid tokens been over cached. This works, but we have another elegant way.
        #score = score + 0.0 * torch.rand_like(score, device= score.device)
    elif cache_dic['cache_type'] == 'kv-norm':
        score = kv_norm_score(cache_dic, current)

    elif cache_dic['cache_type'] == 'similarity':
        # why don't we calculate similarity score? 
        # This is natural but we find it cost **TOO MUCH TIME**, cause in DiT series models, you can calculate similarity for scoring every where.
        score = similarity_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'norm':
        # an interesting exploration, but not used in the final version.
        # use norm as the selectioon method is probably because of the norm of the tokens may indicate the importance of the token. but it is not the case.
        score = norm_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'compress':
        # if you want to combine any of the methods mentioned, we have not tried this yet hhh.
        score1 = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1])
        score1 = torch.cat([score1, score1], dim=0).to(tokens.device)
        score2 = cache_dic['attn_map'][-1][current['layer']].sum(dim=1)#.mean(dim=0) # (B, N)
        # normalize
        score2 = score2 / score2.max(dim=1, keepdim=True)[0]
        score = 0.5 * score1 + 0.5 * score2

    # abandon the branch, if you want to explore the local force fresh strategy, this may help.
    #if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')): # current['is_force_fresh'] is False, cause when it is True, no cut and fresh are needed
    #        #print(torch.ones_like(force_indices, dtype=float, device=force_indices.device).dtype)
    #    score.scatter_(dim=1, index=force_indices, src=torch.ones_like(force_indices, dtype=torch.float32, 
    #                                                                       device=force_indices.device))
    
    if (True and (cache_dic['force_fresh'] == 'global')):
        # apply s3 mentioned in toca, the "True" above is for a switch to turn on/off the s3.
        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['interval'])

        # layer wise s3, not used in the final version. seems it is not necessary to add if step wise is applied.
        #soft_layer_score = cache_dic['cache_index']['layer_index'][current['module']].float() / (27)
        score = score + cache_dic['soft_fresh_weight'] * soft_step_score #+ 0.1 *soft_layer_score
    
    #cfg_score, no_cfg_score = torch.split(score, len(score)//2, dim = 0)
    #score = 0.5* cfg_score + 0.5* no_cfg_score
    #score = torch.cat([score,score], dim=0)

    return score.to(tokens.device)

import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_score(cache_dic, current):
    '''
    Attention Score s1 (s2, but dit doesn't contain cross-attention for s2)
    '''
    #self_attn_score = 1- cache_dic['attn_map'][-1][current['layer']].diagonal(dim1=1, dim2=2)
    #self_attn_score = F.normalize(self_attn_score, dim=1, p=2)
    attention_score = F.normalize(cache_dic['attn_map'][-1][current['layer']].sum(dim=1), dim=1, p=2)
    #score = self_attn_score
    score = attention_score
    return score

def similarity_score(cache_dic, current, tokens):
    cosine_sim = F.cosine_similarity(tokens, cache_dic['cache'][-1][current['layer']][current['module']], dim=-1)

    return F.normalize(1- cosine_sim, dim=-1, p=2)

def norm_score(cache_dic, current, tokens):
    norm = tokens.norm(dim=-1, p=2)
    return F.normalize(norm, dim=-1, p=2)

def kv_norm_score(cache_dic, current):
    # (B, num_heads, N)
    #k_norm = cache_dic['cache'][-1][current['layer']]['k_norm']
    v_norm = cache_dic['cache'][-1][current['layer']]['v_norm']
    kv_norm = 1- v_norm 


    return F.normalize(kv_norm.sum(dim = -2), p=2)


import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    # Update the cached tokens at the positions
    if module == 'attn': 
        # this branch is not used in the final version, but if you explore the partial fresh strategy of attention, it works.
        indices = fresh_indices.sort(dim=1, descending=False)[0]
        
        cache_dic['attn_map'][-1][layer].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_attn_map.shape[-1]), src=fresh_attn_map)
    elif module == 'mlp':
        indices = fresh_indices

    cache_dic['cache'][-1][layer][module].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
            
    

        
        
import re
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_dpmsolver_multistep import \
#     DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
#from models.rdt.model import RDT


import models.rdt_diffuser.schedulers.scheduling_ddpm  as ddpm_mod
import models.rdt_diffuser.schedulers.scheduling_dpmsolver_multistep as dpms_mod
from models.rdt_img_p import model
from models.img_token_prune import overlay_grid,overlay_prune_mask,debug_visualize
from models.img_token_prune import compute_attn_score_with_rdt_head, prune_tokens_with_scores

from models.img_token_prune import prune_tokens_random
import inspect
from colorama import init
from termcolor import colored
from PIL import Image, ImageDraw



class RDTRunner_img_p(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(RDTRunner_img_p, self).__init__()
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.model = model.RDT_img_p(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim * 2,    # state + state mask (indicator)
            out_features=hidden_size
        )
        
        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = ddpm_mod.DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = dpms_mod.DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.vis_count=0
        # print("Diffusion params: %e" % sum(
        #     [p.numel() for p in self.model.parameters()] + 
        #     [p.numel() for p in self.lang_adaptor.parameters()] + 
        #     [p.numel() for p in self.img_adaptor.parameters()] + 
        #     [p.numel() for p in self.state_adaptor.parameters()]))
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs,
                           add_img_pos,
                           img_keep_idx):
                           
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # Set step values
        #flag
        #self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        #print("self.num_inference_timesteps: ",self.num_inference_timesteps)
        self.noise_scheduler_sample.set_timesteps(5)
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            # model_output = self.model(state_action_traj, ctrl_freqs,
            #                         t.unsqueeze(-1).to(device),
            #                         lang_cond, img_cond, lang_mask=lang_attn_mask)
            model_output = self.model(state_action_traj, ctrl_freqs,
                                 t.unsqueeze(-1).to(device),
                                 lang_cond, img_cond,
                                 lang_mask=lang_attn_mask,
                                 img_mask=None,
                                 add_img_pos=add_img_pos,
                                 img_keep_idx=img_keep_idx)
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs
                    ) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        # Align the dimension with the hidden size
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)
        # Predict the denoised result
        pred = self.model(state_action_traj, ctrl_freqs, 
                          timesteps, lang_cond, img_cond, 
                          lang_mask=lang_attn_mask)

        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        return loss
    
    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs,
                       raw_images: List[Image.Image] = None,
                       alpha=0,temp=0,keep_img_token_nums=0,
                       pruning_mode="",save_vis_doc_name=""):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)
        
        # print(f"lan L  : "
        #     + colored(f"{lang_cond.shape}", "yellow"))
        # print(f"img L  : "
        #     + colored(f"{img_cond.shape}", "yellow"))
        
        # print("before token pruning")
        # print(f"before token pruning: "
        #     + colored(f"{img_cond.shape}", "red"))
        # ====== 轻量 cross-attn 打分（复用 RDT 第一层权重） ======

        
        pruned_views, keep_idx_views, attn_grids, sim_matrices = [], [], [], []
        cross_attn0 = self.model.blocks[0].cross_attn
        num_views=6
        B, total_L, D = img_cond.shape
        L = total_L // num_views
        img_view = img_cond.view(B, num_views, L, D)
        tokens_per_view = total_L // num_views
        grid = int(tokens_per_view ** 0.5)
        grid_rows = grid_cols = grid
        
        
        
        for v in range(num_views):
            tokens_v = img_view[:, v, :, :]                 # (1, L, D)
            if pruning_mode == "score":
                attn_score_v = compute_attn_score_with_rdt_head(
                    text_cond=lang_cond,              # (B, L_txt, D)
                    img_cond=tokens_v,                # (B, L_img, D) — 这里 tokens_v=(1, L, D)
                    cross_attn_module=cross_attn0
                ).to(torch.float32).cpu().squeeze(0).reshape(grid_rows, grid_cols).numpy()
                # 2) 自相似度 (L,L)
                tv = tokens_v.squeeze(0)  # (L, D)
                sim_v = F.cosine_similarity(
                    tv.unsqueeze(1), tv.unsqueeze(0), dim=-1
                ).to(torch.float32).cpu().numpy()
                # 3) 评分剪枝
                pruned_v, keep_idx_v = prune_tokens_with_scores(
                    tokens=tokens_v,
                    attn_score=attn_score_v.flatten(),   # 已是 np.ndarray
                    alpha=alpha, sim_temp=temp, topk=keep_img_token_nums
                )
            elif pruning_mode == "random":
                # —— 随机剪枝分支（无需计算打分/相似度）——
                pruned_v, keep_idx_v = prune_tokens_random(tokens=tokens_v, topk=keep_img_token_nums)
                # 若要可视化，随便填个占位（全 0 或均匀）：
                attn_score_v = torch.zeros(grid_rows, grid_cols)
                sim_v = torch.eye(L)
            else:
                # 不剪枝（保留全部），可当第三个基线
                keep_idx_v = torch.arange(L, device=tokens_v.device)[None, :L]
                pruned_v = tokens_v
                attn_score_v = torch.zeros(grid_rows, grid_cols)
                sim_v = torch.eye(L)
            pruned_views.append(pruned_v)
            keep_idx_views.append(keep_idx_v.flatten())
            attn_grids.append(attn_score_v)
            sim_matrices.append(sim_v)

        pruned_img_cond = torch.cat(pruned_views, dim=1)      # (1,6*keep_k,D)
        #keep_idxs = [ki.cpu().numpy() for ki in keep_idx_views]   
        #img_keep_idx = torch.stack(keep_idx_views, dim=0)  # shape = (num_views, K)
        # 1. 把 6 路索引拼成一个一维向量，长度 = 6 * keep_k
        flat_keep_idx = torch.cat(keep_idx_views, dim=0)      # shape = (6*keep_k,)

        # 2. 加上 batch 维
        #    如果 batch_size > 1，需要对每个样本重复；这里 B=1
        img_keep_idx = flat_keep_idx.unsqueeze(0)            # shape = (1, 6*keep_k)
        vis_token_and_save=True
        record_conditional_sample=False
        if raw_images is not None and vis_token_and_save:
            self.vis_count=self.vis_count+1
            if self.vis_count<10:
                
                keep_idxs_np = [ki.cpu().numpy().flatten() for ki in keep_idx_views]
                debug_visualize(
                    images=raw_images,
                    grid_rows=grid_rows, grid_cols=grid_cols,
                    keep_idxs=keep_idxs_np,
                    attn_grids=attn_grids,
                    sim_matrices=sim_matrices,
                    save_vis_doc_name=save_vis_doc_name
                )
        

        # print("after token pruning")
        # print(f"after token pruning: "
        #     + colored(f"{pruned_img_cond.shape}", "red"))
        # ====== 同步裁剪原始 image_embeds（如果后续还要用它，比如存视频可选） ======
        # image_embeds_pruned = image_embeds[:, keep_idx[0], :]

        # ====== 同步位置嵌入（绝对PE） ======
        # 在 RDTRunner.predict_action 中做更好，但你也可在此处直接加 pos_embed
        # 推荐：把 keep_idx 传给 predict_action，让里面自己 gather pos_embed。
        
        cls = self.__class__
        # print(f"类对象: "
        #     + colored(f"{cls}", "red"))
        
        if record_conditional_sample:
            torch.cuda.synchronize()  # 等待当前设备上的所有流完成:contentReference[oaicite:0]{index=0}
            start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            print(colored(inspect.getfile(self.conditional_sample),'blue')) # 打印 .py 文件路径 :contentReference[oaicite:1]{index=1}
            # ===== 这里放置你的 GPU 代码片段 =====
            action_pred = self.conditional_sample(
                lang_cond, lang_attn_mask, pruned_img_cond, 
                state_traj, action_mask, ctrl_freqs,
                add_img_pos=True,img_keep_idx=img_keep_idx
            )
            # =====================================
            torch.cuda.synchronize()  # 等待流中所有内核和事件完成:contentReference[oaicite:2]{index=2}
            end.record()
            # 再次同步，确保结束事件已被记录
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)  # 返回单位为 ms:contentReference[oaicite:3]{index=3}
            print(f"self.policy.predict_action GPU time: "
                + colored(f"{elapsed_ms:.3f}", "red")
                + " ms")
        else:
            action_pred = self.conditional_sample(
                lang_cond, lang_attn_mask, pruned_img_cond, 
                state_traj, action_mask, ctrl_freqs,
                add_img_pos=True,img_keep_idx=img_keep_idx
            )
        
        elapsed_ms=0.00
        return action_pred,elapsed_ms
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
#
#from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
#from diffusers.schedulers.scheduling_dpmsolver_multistep import \
#    DPMSolverMultistepScheduler

from PIL import Image, ImageDraw
#from models.rdt_diffuser.schedulers.scheduling_ddpm import DDPMScheduler
#from models.rdt_diffuser.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
import models.rdt_diffuser.schedulers.scheduling_ddpm  as ddpm_mod
import models.rdt_diffuser.schedulers.scheduling_dpmsolver_multistep as dpms_mod


from  models.taylor.cache_functions.cache_init import cache_init



#img token p
from models.img_token_prune import overlay_grid,overlay_prune_mask,debug_visualize
from models.img_token_prune import compute_attn_score_with_rdt_head, prune_tokens_with_scores
from models.img_token_prune import prune_tokens_random

from PIL import Image, ImageDraw
from utils.profiler import global_profiler 
from utils.cdpruner import prune_tokens_with_cdp,prune_tokens_with_cdp_V2,visualize_cdp_pruning

from utils.profiler import global_class_print_count, PRINT_LIMIT
#验证来源
# print(ddpm_mod)global_class_print_count
# print(dpms_mod)

# # 方式二：查看模块的 __file__ 属性
# print("DDPMScheduler module file:", ddpm_mod.__file__)
# print("DPMSolverMultistepScheduler module file:", dpms_mod.__file__)

# # 方式三：使用 inspect.getfile（效果同 __file__）
# import inspect
# print("DDPMScheduler via inspect:", inspect.getfile(ddpm_mod))
# print("DPMSolverMultistepScheduler via inspect:", inspect.getfile(dpms_mod))

from .project_enums import PruningMode 

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT
#from models.rdt_taylor import model 
from models.rdt_taylor_img_p import model 

from typing import List, Dict
import numpy as np
import statistics
def _build_step_policy(policy):
    sched = []
    for tag in policy:
        tag = tag.upper()
        if tag == 'F':
            sched.append(('full', None))
        elif tag.startswith('T'):
            try:
                order = int(tag[1:])
            except:
                raise ValueError(f'Bad tag: {tag}')
            sched.append(('taylor', order))
        else:
            raise ValueError(f'Unknown policy tag: {tag}')
    return sched
class lzw_stats:
    def __init__(self, mean: float, std: float, nums: int):
        self.mean = mean
        self.std = std
        self.nums = nums

    def __repr__(self):
        return f"Stats(mean={self.mean:.3f}, std={self.std:.3f}, nums={self.nums})"
class RDTRunner_taylor_img_p(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(RDTRunner_taylor_img_p, self).__init__()
        self.vis_count=10
        # ===== TaylorSeer 推理超参，手写 =====
        self.taylor_kwargs = dict(
            interval   = 1,     # 每隔 4 步做一次完整前向
            max_order  = 1,     # 泰勒展开最高阶
            test_FLOPs = False  # 若想统计 FLOPs 改 True
        )
        self.time_records = {
            'full-sa': [], 'full-ca': [], 'full-mlp': [],
            'taylor-sa': [], 'taylor-ca': [], 'taylor-mlp': [],
            'flayer': []
        }
        
        self.one_step_diffusion_record_time:Dict[str, lzw_stats] = {
            k: None for k in self.time_records.keys()
            }
        
        self.one_diffusion_record_time:Dict[str, lzw_stats] = {
            k: None for k in self.time_records.keys()
            }
        
        self.all_diffusion_record_time = {k: [] for k in self.time_records}
        
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        
        
        # self.model = RDT(
        #     output_dim=action_dim,
        #     horizon=pred_horizon,
        #     hidden_size=hidden_size,
        #     depth=config['rdt']['depth'],
        #     num_heads=config['rdt']['num_heads'],
        #     max_lang_cond_len=max_lang_cond_len,
        #     img_cond_len=img_cond_len,
        #     lang_pos_embed_config=lang_pos_embed_config,
        #     img_pos_embed_config=img_pos_embed_config,
        #     dtype=dtype,
        # )
        
        self.model = model.RDT_taylor_img_p(
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
            time_records=self.time_records
        )
        
        print("using taylor")
        
        print(self.model)
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

        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] + 
            [p.numel() for p in self.lang_adaptor.parameters()] + 
            [p.numel() for p in self.img_adaptor.parameters()] + 
            [p.numel() for p in self.state_adaptor.parameters()]))
    
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
    #flag
    def conditional_sample_taylor(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs,
                           img_keep_idx=None):
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
    

        #self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        #num_inference_timesteps=5
        self.noise_scheduler_sample.set_timesteps(5)
        timesteps = self.noise_scheduler_sample.timesteps  # [999, 799, …, 200]
        a=self.num_inference_timesteps
        
        
        
        policy = getattr(self, 'sample_policy', None)
        if policy is None:
            policy = ['F','T0','F','T1','F']  # 默认
        schedule = _build_step_policy(policy)

        # 用策略长度决定采样步数（保持和策略一致）
        self.noise_scheduler_sample.set_timesteps(len(schedule))
        timesteps = self.noise_scheduler_sample.timesteps  # 长度 == len(schedule)

        # 初始化缓存
        cache_dic, current = cache_init(model_kwargs=self.taylor_kwargs,
                                        num_steps=len(schedule),
                                        num_layers=self.model.depth)
        current['activated_steps'] = []  # 存 Full 的 t
        #print(schedule)
        for step_idx, t in enumerate(timesteps):
            
            global_profiler.start_diffusion_step(step_idx=step_idx) # <--- 在这里调用
            #print("diffusion step idx:",step_idx," t: ",t)
            current['step_idx'] = step_idx
            current['step']     = float(t.item())

            # ----------- 外层决定模式 -----------
            mode, order = schedule[step_idx]
            if mode == 'full':
                current['type'] = 'full'
                current['taylor_order'] = None
                current['activated_steps'].append(current['step'])  # 只在 Full 记录锚点
            else:
                current['type'] = 'taylor'
                current['taylor_order'] = int(order)  # 例如 0 / 1 / 2 ..
            #print("current['type']:",current['type'] )
            # -----------------------------------
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            # def forward(self, current ,cache_dic ,
            #     x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None): 
            
            
            
            
            # torch.cuda.synchronize()  # 等待当前设备上的所有流完成:contentReference[oaicite:0]{index=0}
            model_start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
            model_end   = torch.cuda.Event(enable_timing=True)
            model_start.record()

            # ===== 这里放置你的 GPU 代码片段 =====
            model_output = self.model( current ,cache_dic ,state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    lang_cond, img_cond, lang_mask=lang_attn_mask,
                                    img_keep_idx=img_keep_idx) # <-- 传递给 RDT 模型
            # =====================================
            # torch.cuda.synchronize()  # 等待流中所有内核和事件完成:contentReference[oaicite:2]{index=2}
            model_end.record()
            # # 再次同步，确保结束事件已被记录
            torch.cuda.synchronize()
            model_time = model_start.elapsed_time(model_end)  # 返回单位为 ms:contentReference[oaicite:3]{index=3}
            print("-------------------------------")
            print("model running time: ",model_time)
            print("-------------------------------")
        
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
            #
            #print
        #     for key, lst in self.time_records.items():
        #         if not lst:  # 若某个模块根本没有数据，可跳过或填 0
        #             self.one_step_diffusion_record_time[key] = lzw_stats(0.0, 0.0, 0)
        #             continue
        #         mean_val = statistics.mean(lst)             # 算术平均
        #         std_val  = statistics.pstdev(lst)           # 总体标准差
        #         cnt      = len(lst)                         # 样本数量
        #         self.one_step_diffusion_record_time[key] = lzw_stats(
        #             mean=mean_val, std=std_val, nums=cnt
        #         )
        #     for name, stats in self.one_step_diffusion_record_time.items():
        #         print(f"{name:12s} -> {stats}")
                
        #     #clear one step diffusion record time
        #     for k, lst in self.time_records.items():
        #         lst.clear()
            
        #     # 3) 累积每步的 mean
        #     for key, stats in self.one_step_diffusion_record_time.items():
        #         self.all_diffusion_record_time[key].append(stats.mean)
        # #print("begin to calculate mean of mean")
        # self.final_diffusion_stats = {}
        # for key, means in self.all_diffusion_record_time.items():
        #     if means:
        #         overall_mean = statistics.mean(means)
        #         overall_std  = statistics.pstdev(means)
        #         cnt = len(means)
        #     else:
        #         overall_mean = overall_std = 0.0
        #         cnt = 0
        #     self.final_diffusion_stats[key] = lzw_stats(overall_mean, overall_std, cnt)

        # 打印 overall stats
        # for name, stats in self.final_diffusion_stats.items():
        #     print(f"{name:12s} -> {stats}")    
            
            
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    
    
    
    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
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
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            model_output = self.model(state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    lang_cond, img_cond, lang_mask=lang_attn_mask)
            
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
    # def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
    #                    action_mask, ctrl_freqs):
    #     '''
    #     lang_tokens: (batch_size, lang_len, lang_token_dim)
    #     lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
    #         which should be True-False bool tensor.
    #     img_tokens: (batch_size, img_len, img_token_dim)
    #     state_tokens: (batch_size, 1, state_token_dim)
    #     action_mask: (batch_size, 1, action_dim),
    #         which should be a 0-1 **float** tensor.
    #     ctrl_freqs: (batch_size,), control frequency for each sample.
        
    #     return: (batch_size, horizon, action_dim), predicted action sequence
    #     '''
    #     # Prepare the state and conditions
    #     state_tokens = torch.cat([state_tokens, action_mask], dim=2)
    #     lang_cond, img_cond, state_traj = self.adapt_conditions(
    #         lang_tokens, img_tokens, state_tokens)
        
    #     # Run sampling
    #     #flag
    #     action_pred = self.conditional_sample_taylor(
    #         lang_cond, lang_attn_mask, img_cond, 
    #         state_traj, action_mask, ctrl_freqs,
    #     )
        
    #     return action_pred
    # def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
    #                     action_mask, ctrl_freqs):
        
        
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs,
                       raw_images: List[Image.Image] = None,
                       alpha=0,temp=0,keep_img_token_nums=0,
                       theta=0,
                       # 2. 修改这里的类型提示，从 str 变为 PruningMode
                       pruning_mode: PruningMode = PruningMode.NO_PRUNING, # 可以给个默认值
                       
                       save_vis_doc_name=""):
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
        # 3. (可选但推荐) 在函数开始时进行类型检查，确保传入的是正确的枚举类型
        
        # #timing flag
        # preprocess_start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
        # preprocess_end   = torch.cuda.Event(enable_timing=True)
        # preprocess_start.record()
        # #timing flag
        
        if not isinstance(pruning_mode, PruningMode):
            raise TypeError(f"pruning_mode 必须是 PruningMode 枚举，但收到了 {type(pruning_mode)}")

        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)  
        print("before pruning: ")
        print(f"lang_cond:  shape={lang_cond.shape}, num_elements={lang_cond.numel()}")
        print(f"img_cond:   shape={img_cond.shape}, num_elements={img_cond.numel()}")
        print(f"state_traj: shape={state_traj.shape}, num_elements={state_traj.numel()}")
        
        
        cls = self.__class__
        # 使用全局计数器来控制打印
        global global_class_print_count
        
        import inspect
        from colorama import init
        from termcolor import colored
        from utils.profiler import global_class_print_count
        global_class_print_flag=False
        
        #if global_class_print_count<=PRINT_LIMIT: 
        if global_class_print_flag:
            print(colored(inspect.getfile(self.conditional_sample_taylor),'blue')) # 打印 .py 文件路径 :contentReference[oaicite:1]{index=1}
            global_class_print_count=global_class_print_count+1
            print("类对象:", cls)
            print("global_class_print_count:", global_class_print_count)
            
        
        
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
        
        #创建一个专门存放【局部索引】的列表，用于可视化
        local_keep_indices_for_vis = []   
        # #timing flag
        
        # preprocess_end.record()
        # torch.cuda.synchronize()
        # preprocess_time = preprocess_start.elapsed_time(preprocess_end)
        # print("preprocessg time : ",preprocess_time)
        # #timing flag
        
        
        
        # #timing flag
        # pruning_start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
        # pruning_end   = torch.cuda.Event(enable_timing=True)
        # pruning_start.record()
        # #timing flag
        for v in range(num_views):
            tokens_v = img_view[:, v, :, :]                 # (1, L, D)
            #if pruning_mode == "score":
            if pruning_mode == PruningMode.SCORE:
                
                # #timing flag
                # single_pruning_start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
                # single_pruning_end   = torch.cuda.Event(enable_timing=True)
                # single_pruning_start.record()
                # #timing flag
                
                
                # 1. 计算注意力分数 (保持不变)
                attn_score_v = compute_attn_score_with_rdt_head(
                    text_cond=lang_cond,
                    img_cond=tokens_v,
                    cross_attn_module=cross_attn0
                ).to(torch.float32) # 返回 (B, L) 的Tensor
                # 2. 【删除】不再在这里计算相似度矩阵
                # tv = tokens_v.squeeze(0)
                # sim_v = F.cosine_similarity(...) # <--- 删除此行
                # 3. 【修正】调用GPU优化后的剪枝函数，只传递必需的参数
                #    prune_tokens_with_scores 函数会自己处理相似度计算
                pruned_v, keep_idx_v ,sim_v = prune_tokens_with_scores(
                    tokens=tokens_v,
                    attn_score=attn_score_v, # 直接传递 (B, L) 的Tensor
                    alpha=alpha, 
                    sim_temp=temp, 
                    topk=keep_img_token_nums
                )
                
                # #timing flag
                # single_pruning_end.record()
                # torch.cuda.synchronize()
                # single_pruning_time = single_pruning_start.elapsed_time(single_pruning_end)
                # print("single_pruning time : ",single_pruning_time)
                # #timing flag
                
                
                
                
                # 为了可视化，我们需要将 sim_v 和 attn_score_v 传递出去
                # 我们可以在剪枝函数中让它返回这些中间结果，或者在这里重新计算一次（仅为了可视化）
                # 为了代码清晰，我们暂时只关注剪枝逻辑
            #elif pruning_mode == "random":
            elif pruning_mode == PruningMode.RANDOM:
                # —— 随机剪枝分支（无需计算打分/相似度）——
                pruned_v, keep_idx_v = prune_tokens_random(tokens=tokens_v, topk=keep_img_token_nums)
                # 若要可视化，随便填个占位（全 0 或均匀）：
                attn_score_v = torch.zeros(grid_rows, grid_cols)
                sim_v = torch.eye(L)
            # =================== 新增 CDPRUNING 分支 ===================
            elif pruning_mode == PruningMode.CDPRUNER:
                
                global cdp_para
                # prune_tokens_with_cdp 返回的是相对于当前视图 v 的【局部索引】
                # 它的值范围永远是 [0, 728]
                pruned_v, keep_idx_v = prune_tokens_with_cdp_V2(
                    tokens=tokens_v,
                    lang_cond=lang_cond, # 传入语言条件
                    topk=keep_img_token_nums,
                    theta=theta
                )
                # CDP的可视化比较复杂，这里暂时跳过
                # 【新增】为可视化创建占位符
                attn_score_v = torch.zeros(L, device=tokens_v.device)
                sim_v = torch.eye(L, device=tokens_v.device)
                # 1. 保存【局部索引】用于可视化 直接将原始的局部索引存入A列表
                local_keep_indices_for_vis.append(keep_idx_v.squeeze(0)) 
            # ==========================================================
            #elif pruning_mode=="no_img_token_pruning":
            elif pruning_mode == PruningMode.NO_PRUNING:
                # 不剪枝（保留全部），可当第三个基线
                keep_idx_v = torch.arange(L, device=tokens_v.device)[None, :L]
                pruned_v = tokens_v
                attn_score_v = torch.zeros(grid_rows, grid_cols)
                sim_v = torch.eye(L)
                
            # 关键步骤：将局部索引转换为全局索引
            offset = v * L 
            global_keep_idx_v = keep_idx_v + offset
            #offset
            
            pruned_views.append(pruned_v)
            keep_idx_views.append(global_keep_idx_v)
            attn_grids.append(attn_score_v)
            sim_matrices.append(sim_v)

        
        # #timing flag
        # pruning_end.record()
        # torch.cuda.synchronize()
        # pruning_time = pruning_start.elapsed_time(pruning_end)
        # print("pruning time : ",pruning_time)
        # #timing flag
        
        
        #offseet
        pruned_img_cond = torch.cat(pruned_views, dim=1)
        img_keep_idx = torch.cat(keep_idx_views, dim=1) # (B, 6*keep_k)
        
        #offset
        
        print("after pruning: ")
        print(f"lang_cond:  shape={lang_cond.shape}, num_elements={lang_cond.numel()}")
        print(f"img_cond:   shape={pruned_img_cond.shape}, num_elements={pruned_img_cond.numel()}")
        print(f"state_traj: shape={state_traj.shape}, num_elements={state_traj.numel()}")
        
        
        #vis for score 
        vis_token_and_save=False
        visualize_cdp=False
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
        

        if pruning_mode == PruningMode.CDPRUNER and visualize_cdp:
            

            visualize_cdp_pruning(
                images=raw_images,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                keep_idxs_per_view=local_keep_indices_for_vis,
                save_vis_doc_name=save_vis_doc_name
            )
    
        # print("after token pruning")
        # print(f"after token pruning: "
        #     + colored(f"{pruned_img_cond.shape}", "red"))
        
        
        
        
        
        # torch.cuda.synchronize()  # 等待当前设备上的所有流完成:contentReference[oaicite:0]{index=0}
        # start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
        # end   = torch.cuda.Event(enable_timing=True)
        # start.record()
        
        
        # ===== 这里放置你的 GPU 代码片段 =====
        action_pred = self.conditional_sample_taylor(
            lang_cond, lang_attn_mask, pruned_img_cond, 
            state_traj, action_mask, ctrl_freqs,
            img_keep_idx=img_keep_idx # <-- 传递保留的索引
        )
        # =====================================
        # torch.cuda.synchronize()  # 等待流中所有内核和事件完成:contentReference[oaicite:2]{index=2}
        # end.record()
        # # 再次同步，确保结束事件已被记录
        # torch.cuda.synchronize()
        # elapsed_ms = start.elapsed_time(end)  # 返回单位为 ms:contentReference[oaicite:3]{index=3}
        # print(f"self.policy.predict_action GPU time: "
        #     + colored(f"{elapsed_ms:.3f}", "red")
        #     + " ms")
        # action_pred = self.conditional_sample(
        #     lang_cond, lang_attn_mask, img_cond, 
        #     state_traj, action_mask, ctrl_freqs,
        # )
        elapsed_ms=0
        return action_pred,elapsed_ms
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)



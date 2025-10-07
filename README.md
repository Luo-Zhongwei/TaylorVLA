# TaylorVLA a light Diffusion Foundation Model for Bimanual Manipulation

### 📝[Paper](https://arxiv.org/pdf/2410.07864) | 🌍[Project Page](https://rdt-robotics.github.io/rdt-robotics/) | 🤗[Model](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | 🛢️[Data](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data) | 🏞️[Poster](./assets/iclr2025_poster.png)

![](./assets/head.png)

RDT-1B is a **1B**-parameter (*largest* to date) imitation learning **Diffusion Transformer** pre-trained on **1M+** (*largest* to date) multi-robot episodes. Given language instruction and RGB images of up to three views, RDT can predict the next $64$ robot actions. RDT is inherently compatible with **almost all kinds of modern mobile manipulators**, from single-arm to dual-arm, joint to EEF, position to velocity, and even with wheeled locomotion.

We have fine-tuned RDT on **6K+** (one of the *largest*) self-collected bimanual episodes and deployed it on the ALOHA **dual-arm** robot. It has achieved state-of-the-art performance in terms of dexterity, zero-shot generalizability, and few-shot learning. You can find Demo videos on our [project page](https://rdt-robotics.github.io/rdt-robotics/).

This repo is an official PyTorch implementation of RDT, containing:

- 🛠️Model [implementation](models/rdt_runner.py) of RDT
- 🤗1M-step [checkpoint](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) of RDT-1B pre-trained on multi-robot data
- 🤗500K-step [checkpoint](https://huggingface.co/robotics-diffusion-transformer/rdt-170m) of RDT-170M (RDT(small) in [ablation](https://arxiv.org/pdf/2410.07864))
- 📈Training and sampling [scripts](train/train.py) (with DeepSpeed)
- 🤖An [example](scripts/agilex_inference.py) of real-robot deployment
- 🕹️Simulation benchmark from [Maniskill](https://github.com/haosulab/ManiSkill) environment

The following guides include the [installation](#installation), [fine-tuning](#fine-tuning-on-your-own-dataset), and [deployment](#deployment-on-real-robots). Please refer to [pre-training](docs/pretrain.md) for a detailed list of pre-training datasets and a pre-training guide.

## 📰 News
- [2025/04/04] [Poster](./assets/iclr2025_poster.png) is uploaded.
- [2024/12/17] 🔥 [Scripts](#simulation-benchmark) for evaluating RDT in Maniskill Simulation Benchmark is released!
- [2024/10/23] 🔥 **RDT-170M** (Smaller) model is released, a more VRAM-friendly solution 🚀💻.

## Installation

1. Clone this repo and install prerequisites:

    ```bash
    
    ```

2. Download off-the-shelf multi-modal encoders:

   You can download the encoders from the following links:

   - `t5-v1_1-xxl`: [link](https://huggingface.co/google/t5-v1_1-xxl/tree/main)🤗
   - `siglip`: [link](https://huggingface.co/google/siglip-so400m-patch14-384)🤗

   And link the encoders to the repo directory:

   ```bash
   # Under the root directory of this repo
   mkdir -p google
   
   # Link the downloaded encoders to this repo
   ln -s /path/to/t5-v1_1-xxl google/t5-v1_1-xxl
   ln -s /path/to/siglip-so400m-patch14-384 google/siglip-so400m-patch14-384
   ```
3. Fill the missing argument in [this file](configs/base.yaml#L22):
   
   Note that this buffer will only be used during pre-training. See [this doc](docs/pretrain.md) for more details.
   ```
   # ...
   
   dataset:
   # ...
   # ADD YOUR buf_path: the path to the buffer (at least 400GB)
      buf_path: /path/to/buffer
   # ...
   ```


## Deployment on Real-Robots

We have encapsulated the inference of the model into a class named `RoboticDiffusionTransformerModel` (see [this file](scripts/agilex_model.py#L38)). You can call this class's `step()` method for inference. However, you may need to re-implement some parts according to your specific robot. You should at least modify the `_format_joint_to_state()` (L164) and `_unformat_action_to_joint()` (L196) to convert between robot raw actions and unified action vectors that RDT accepts. You may also specify the control frequency of your robot (L49).

**IMPORTANT**: When you feed the images into `step()`, remember the order MUST be `[ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, ext_{t}, right_wrist_{t}, left_wrist_{t}]`.

We provide an example hardware code in [this file](scripts/agilex_inference.py) for deployment on Mobile ALOHA, and the corresponding running script in [this file](inference.sh) (`inference.sh`), which is detailed below;

   ```bash
      python -m scripts.agilex_inference \
         --use_actions_interpolation \
         --pretrained_model_name_or_path=<PATH TO MODEL CHECKPOINT> \  # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt, the same before
         --lang_embeddings_path=<PATH TO YOUR INSTURCTION EMBEDDINGS> \ # e.g. outs/lang_embeddings/your_instr.pt"
         --ctrl_freq=25    # your control frequency
   ```

**IMPORTANT**: If you on-board GPU memory is not enough to encode the language, please refer to [this file](scripts/encode_lang.py) for precomputation and specify the language embedding path in `inference.sh`. Detail instructions are provided below:

   1. Set Required Parameters in `scripts/encode_lang.py`

      ```python
      # ...

      GPU = 0
      MODEL_PATH = "google/t5-v1_1-xxl"
      CONFIG_PATH = "configs/base.yaml"
      SAVE_DIR = "outs/"   # output directory

      # Modify this to your task name and instruction
      TASK_NAME = "handover_pan"
      INSTRUCTION = "Pick up the black marker on the right and put it into the packaging box on the left."

      # Note: if your GPU VRAM is less than 24GB, 
      # it is recommended to enable offloading by specifying an offload directory. 
      OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

      # ...
      ```

   2. Run the script
      ```
      python -m scripts.encode_lang
      ```

Note: If you want to deploy on the Mobile ALOHA robot, don't forget to install the hardware prerequisites (see [this repo](https://github.com/MarkFzp/mobile-aloha)).

## Simulation Benchmark

We comprehensively evaluate RDT against baseline methods using the ManiSkill simulation benchmark. Specifically, we focus on five benchmark tasks: `PegInsertionSide`, `PickCube`, `StackCube`, `PlugCharger`, and `PushCube`. Here's a brief overview of the evaluation setup:

**Evaluation Setup:**

1. **Install ManiSkill:**  
   Within the [RDT environment](#installation), install ManiSkill as follows:
   ```bash
   conda activate rdt
   pip install --upgrade mani_skill
   ```

2. **Configure Vulkan:**  
   Follow the [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to properly set up Vulkan。

3. **Obtain Model Weights:**  
   Download the fine-tuned model weights from [this Hugging Face repository](https://huggingface.co/robotics-diffusion-transformer/maniskill-model/tree/main/rdt). Download the precomputed language embeddings from [here](https://huggingface.co/robotics-diffusion-transformer/maniskill-model/tree/main/lang_embeds) to the root directory of this repo.
   
4. **Run Evaluation Scripts:**  
   After completing the setup steps, execute the provided evaluation scripts to assess RDT on the selected tasks.

```
conda activate rdt 
python -m eval_sim.eval_rdt_maniskill \
--pretrained_path PATH_TO_PRETRAINED_MODEL
```

### Implementation Details

#### Data

Utilizing the [official ManiSkill repository](https://github.com/haosulab/ManiSkill), we generated 5,000 trajectories through motion planning. The initial action mode of these trajectories is absolute joint position control and we subsequently converted them into delta end-effector pose control to align with the pre-training action space of OpenVLA and Octo. We strictly adhered to the official codebases of OpenVLA and Octo, modifying only the dataset-loading scripts. Consequently, we finetuned OpenVLA and Octo using the delta end-effector pose data. For RDT and Diffusion-Policy we leverage joint position control data for training which is aligned with our pre-training stage as well.

####  Training
- OpenVLA is fine-tuned from the officially released pre-trained checkpoint with LoRA-rank 32 until converge.
- Octo is fine-tuned from the officially released pre-trained checkpoint for 1M iterations until converge. 
- Diffusion-Policy is trained from scratch for 1000 epochs. We select the checkpoint of 700 epoch which has the lowest validation sample loss of 1e-3.
- RDT is fine-tuned from our released pre-trained checkpoint for 300ks iterations.

#### Results

Each method is evaluated over 250 trials (10 random seeds with 25 trials per seed). The quantitative results, including success rate mean and std value across 10 random seeds are presented below:


||PegInsertionSide|PickCube|StackCube|PlugCharger|PushCube|Mean|
|---|---|---|---|---|---|---|
|RDT|**13.2±0.29%**|**77.2±0.48%**|74.0±0.30%|**1.2±0.07%**|**100±0.00%**|**53.6±0.52%**|
|OpenVLA|0.0±0.00%|8±0.00%|8±0.00%|0.0±0.00%|8±0.00%|4.8±0.00%|
|Octo|0.0±0.00%|0.0±0.00%|0.0±0.00%|0.0±0.00%|0.0±0.00%|0.0±0.00%|
|Diffusion-Policy|0.0±0.00%|40.0±0.00%|**80.0±0.00%**|0.0%±0.00%|88.0±0.00%|30.2±0.00%|

#### Finetune RDT with Maniskill Data

To fine-tune RDT with Maniskill data, first download the Maniskill data from [here](https://huggingface.co/robotics-diffusion-transformer/maniskill-model) and extract it to `data/datasets/rdt-ft-data`. Then copy the code in `data/hdf5_vla_dataset.py` to `data/hdf5_maniskill_dataset.py` and run the following script:

```
bash finetune_maniskill.sh
```

#### Reproducing Baseline Results

Download and extract the fine-tuned model weights from [here](https://huggingface.co/robotics-diffusion-transformer/maniskill-model) to `eval_sim/`.

- OpenVLA: Clone [OpenVLA repo](https://github.com/openvla/openvla) in `./eval_sim/` and install its environment & ManiSkill. Then run the following script:
```
python -m eval_sim.eval_openvla --pretrained_path PATH_TO_PRETRAINED_MODEL
```
- Octo: Clone [Octo repo](https://github.com/octo-models/octo.git) in `./eval_sim/` and install its environment & ManiSkill. The run the following script:
```
python -m eval_sim.eval_octo --pretrained_path PATH_TO_PRETRAINED_MODEL
```
- Diffusion-Policy: Clone our simplified [Diffusion-Policy repo](https://github.com/LBG21/RDT-Eval-Diffusion-Policy) in `./eval_sim/` and run:
```
python -m eval_sim.eval_dp --pretrained_path PATH_TO_PRETRAINED_MODEL
```

## FAQ

### 1. How can I fine-tune RDTs with limited VRAM?

- **Use a Smaller Model**: Opt for the [RDT-170M model](https://huggingface.co/robotics-diffusion-transformer/rdt-170m), which requires less VRAM.
  
- **Select a Memory-Efficient ZeRO Stage**: Choose a more memory-efficient ZeRO stage based on your needs:
  - **ZeRO-3 with Offload** > **ZeRO-3** > **ZeRO-2 with Offload** > **ZeRO-2** > **ZeRO-1**
  - By default, we use [ZeRO-2](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/c68398ed526733faca4eec52cc1a7d15a9f8fea7/finetune.sh#L29) for a balance between speed and memory efficiency. Find more details on ZeRO stages [here](https://huggingface.co/docs/transformers/main/deepspeed#select-a-zero-stage) and [here](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training).

- **Enable 8-bit Adam Optimization**: Activate 8-bit Adam by setting [`use_8bit_adam=True`](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/c68398ed526733faca4eec52cc1a7d15a9f8fea7/main.py#L195) for reduced memory usage during training.

- **Apply 4-bit or 8-bit Quantization**: Quantizing model weights can significantly reduce VRAM requirements.

- **Use [XFormers](https://github.com/facebookresearch/xformers)**: This library provides optimized transformers with efficient memory usage.

- **Enable Gradient Checkpointing**: Implement `gradient_checkpointing` manually to save memory during backpropagation. See [here](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html) for instructions. Once you have successfully implemented this feature, we welcome you to submit a PR👏.
- **Gradient Accumulation**: Set a larger `--gradient_accumulation_steps=<num_steps>`. This will accumulate the gradients of `<num_steps>` batches for backpropagation. Equivalently, this will increase the batch size by `<num_steps>` times, at the cost of `<num_steps>` times the running time.

### 2. How many steps are recommended for fine-tuning RDT?

Regardless of the batch size you select, it is recommended to train for at least 150K steps to achieve optimal results.

### 3. What to do if t5-xxL is too large to store in GPU memory?

1. Do not load T5-XXL in your GPU memory when training. Pre-compute language embeddings in advance.
2. Set `OFFLOAD_DIR` to enable CPU offloading in `scripts/encode_lang_batch.py` and `scripts/encode_lang.py`.
3. Use smaller versions of t5 like t5-base instead of t5-xxL.

## Citation

If you find our work helpful, please cite us:

```bibtex
@article{liu2024rdt,
  title={RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation},
  author={Liu, Songming and Wu, Lingxuan and Li, Bangguo and Tan, Hengkai and Chen, Huayu and Wang, Zhengyi and Xu, Ke and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2410.07864},
  year={2024}
}
```

Thank you!

## License

All the code, model weights, and data are licensed under [MIT license](./LICENSE).

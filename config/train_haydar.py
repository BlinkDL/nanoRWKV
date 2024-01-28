import git
import torch

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

eval_interval = 100  # keep frequent because we'll overfit
eval_iters = 8
log_interval = 25  # don't print too often

init_from = "scratch"

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# dataset = "chess_small"
# dataset = "lichess_hf_dataset"
# dataset = "shakespeare_char"

dataset = "haydar_dumen"
out_dir = "out-haydar"

gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to tokens

# Model constants
# model_type = "DimensionalGPT"
# model_type = "CustomAttentionGPT"
# model_type = "SilentGPT"
# model_type = "GPT"

n_attention = 3
extra_dim = 8
n_silent = 3
n_layer = 8
n_head = 4
n_embd = 512
dropout = 0.2

learning_rate = 3e-4  # with baby networks can afford to go a bit higher
max_tokens = int(1e9)
lr_decay_iters = 10000
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 10  # not super necessary potentially
compile = False

# Device specific instructions
device = "cuda" if torch.cuda.is_available() else "mps"

# if device == "mps":
#     commit_with_message()

# commit_hash = get_last_commit_hash()
# print(f"current commit commit hash: {commit_hash}")

wandb_log = True  # override via command line if you like
wandb_project = "haydar"
wandb_run_name = f"RWKV"
# wandb_run_name = f"attention_{n_attention}_{elo}_{model_type}_{commit_hash}"
# "dimension",
wandb_tags = [
    "rnn",
    "haydar",
    "rwkv",
]

# train a Shakespeare model from scratch (not character-level)
# good for training on a single GPU with decent performance

out_dir = 'out-shakespeare'
eval_interval = 200  # evaluate somewhat frequently
eval_iters = 200
log_interval = 10

# we might overfit on this relatively small dataset, so only save when validation improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = 'shakespeare'
wandb_run_name = 'shakespeare-from-scratch'

dataset = 'shakespeare'
gradient_accumulation_steps = 4  # increase effective batch size
batch_size = 32
block_size = 256  # context window of 256 tokens

# model configuration - medium sized GPT
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2  # increase dropout as dataset is small

# training
learning_rate = 6e-4  # slightly higher learning rate
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 6e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

# warmup
warmup_iters = 100

# weight decay
weight_decay = 0.1  # add some regularization

# compiler
compile = True  # use PyTorch 2.0 compiler for better performance 
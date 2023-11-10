import json
import torch
import argparse
import subprocess
import logging
import math
import time
import functools

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--no_fsdp",
                    action="store_true",
                    help="Train with DDP instead of FSPD.")
parser.add_argument("--no_layer_wrap_policy",
                    action="store_true",
                    help="Don't use custom FSDP layer wrap policy.")
parser.add_argument("--batch_size_per_device",
                    type=int,
                    default=1,
                    help="Per-device training batch size")
parser.add_argument("--gradient_accumulation_steps",
                    type=int,
                    default=8,
                    help="Gradient accumulation steps")
parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-5,
                    help="Learning Rate")
parser.add_argument("--max_training_steps",
                    type=int,
                    default=-1,
                    help="Interrupt training early.")
parser.add_argument("--gradient_checkpointing",
                    action="store_true",
                    help="Use gradient checkpointing")
parser.add_argument("--max_seq_length",
                    type=int,
                    default=1024,
                    help="Maximum sequence length for truncation.")
args = parser.parse_args()


# Prepare the dataset
class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = [json.loads(line) for line in open(file_path, 'r')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Encoding the text using the tokenizer
        encoded = self.tokenizer(self.data[idx]['text'], return_tensors='pt', truncation=True, max_length=512)
        # We need to return input_ids and attention mask to be used by the model
        return {key: val.squeeze() for key, val in encoded.items()}

    def collate(self, examples):
        max_len = max([len(item['input_ids']) for item in examples])

        input_ids = torch.zeros(len(examples), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(examples), max_len, dtype=torch.long)
        labels = -100 * torch.ones(len(examples), max_len, dtype=torch.long)

        for i, item in enumerate(examples):
            input_ids[i, :len(item['input_ids'])] = item['input_ids']
            attention_mask[i, :len(item['attention_mask'])] = item['attention_mask']
            labels[i, :len(item['input_ids'])] = item['input_ids']

        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)


def train(model, dataset, args):
    sampler = DistributedSampler(dataset,
                                 shuffle=True,
                                 seed=42,
                                 drop_last=True,
                                 num_replicas=torch.distributed.get_world_size(),
                                 rank=torch.distributed.get_rank())
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size_per_device,
                            num_workers=4,
                            collate_fn=dataset.collate,
                            sampler=sampler)

    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Add learning rate schedule with warmup
    num_training_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_warmup_steps = math.ceil(num_training_steps * 0.1)

    # linear learning rate schedule with warmup
    def lr_schedule(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # AMP Setup
    scaler = GradScaler()

    # For demonstration purposes only: Print nvidia-smi GPU interface
    logger.info(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))

    training_step = 0
    loss_training_step = 0

    model.train()

    # Training loop
    start_time = time.time()
    for step, batch in enumerate(dataloader):
        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # AMP autocast
        with autocast(dtype=torch.bfloat16):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss / args.gradient_accumulation_steps

        loss_training_step += loss.cpu().item()

        # Backpropagation with gradient accumulation
        scaler.scale(loss).backward()
        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
            training_step += 1

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            time_elapsed = time.time() - start_time
            start_time = time.time()

            logger.info(f"({training_step}/{num_training_steps}) loss={loss_training_step:.2f} ({time_elapsed:.2f}s/step)")
            loss_training_step = 0
            lr_scheduler.step()

            if training_step == args.max_training_steps:
                return

        torch.cuda.empty_cache()


# Make sure to initialize distributed training (e.g., by using torch.distributed.launch)
torch.distributed.init_process_group(backend="nccl")

# Setup logging
logging.basicConfig(level=(logging.INFO if torch.distributed.get_rank() == 0 else logging.WARNING), format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Initialized process group: {torch.distributed.get_world_size()}")
logger.info(f"Args: {args}")

# Fix seed
torch.random.manual_seed(42)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir=".cache")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir=".cache")
logger.info(model)

torch.cuda.set_device(torch.distributed.get_rank())
device = torch.cuda.current_device()
if args.no_fsdp:
    # Move the model to the GPU and wrap model with DDP (no model sharding)
    model = DDP(model.to(device))
else:
    auto_wrap_policy = None
    if not args.no_layer_wrap_policy:
        # Identify which modules have "layer" in their class name and use these
        # as the basic FSDP blocks that are sharded and exchanged between GPUs
        def layer_policy_fn(module):
            return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=layer_policy_fn)

    # Wrap model as FSDP model
    # If the model is too big, we shouldn't explicitly move it to GPU,
    # but let FSDP handle it.
    model = FSDP(model, device_id=device, auto_wrap_policy=auto_wrap_policy)

# Assuming you have a `data.jsonl` in your current working directory
dataset = JsonlDataset("strategic_game_chess.jsonl", tokenizer)

train(model, dataset, args)
logger.info("Training complete.")

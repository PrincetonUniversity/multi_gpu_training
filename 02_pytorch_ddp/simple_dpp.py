import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(42, 3)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output

rank          = int(os.environ["SLURM_PROCID"])
world_size    = int(os.environ["WORLD_SIZE"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
assert gpus_per_node == torch.cuda.device_count()
print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
      f" {gpus_per_node} allocated GPUs per node.", flush=True)

dist.init_process_group("nccl", rank=rank, world_size=world_size)
if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

local_rank = rank - gpus_per_node * (rank // gpus_per_node)
torch.cuda.set_device(local_rank)

model = Net().to(local_rank)
ddp_model = DDP(model, device_ids=[local_rank])

ddp_model.eval()
with torch.no_grad():
    data = torch.rand(1, 42)
    data = data.to(local_rank)
    output = ddp_model(data)
    print(f"host: {gethostname()}, rank: {rank}, output: {output}")

dist.destroy_process_group()

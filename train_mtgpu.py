import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model.my_model import BertLSTM, BertLSTM1
from data.dataset import NewsDataset
import os
import datetime
from datetime import datetime
from tqdm import tqdm


# def save_and_del(model, epoch, exp_path):
#     torch.save(model.state_dict(), os.path.join(exp_path, f'epoch{epoch}.pth'))
#     if epoch % 5 != 0:
#         last_checkpoint_path = os.path.join(exp_path, f'epoch{epoch-1}.pth')
#         if os.path.exists(last_checkpoint_path):
#             os.remove(last_checkpoint_path)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        exp_path: str,
        save_every: int = 1,
        total_epochs: int = 200,
        check_point: str = None
    ) -> None:
        self.total_epochs = total_epochs
        self.exp_path = exp_path
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])

        if check_point == None:
            self.start_epoch = 0
        else:
            # self.model.load_state_dict(torch.load(check_point)) 
            # self.model = torch.load(check_point)
            self.model.module.load_state_dict(torch.load(check_point))
            self.start_epoch = int(check_point.split('.')[0].split('epoch')[1]) + 1
        self.criterion = nn.CrossEntropyLoss()
        

    def _run_batch(self, input_ids, attention_mask, target_ids) -> float:
        self.optimizer.zero_grad()
        output = self.model(input_ids, attention_mask, target_ids)
        target = torch.zeros_like(output)
        for i in range(output.shape[0]):
            for j in range(target.shape[1]):
                target[i][j][target_ids[i][j]] = 1
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        total_loss = 0.0
        i = 0
        t = tqdm(self.train_data, desc=f"Epoch {epoch}/{self.total_epochs}", leave=False) 
        for input_ids, attention_mask, target_ids in t:
            i += 1
            input_ids = input_ids.to(self.gpu_id)
            attention_mask = attention_mask.to(self.gpu_id)
            target_ids = target_ids.to(self.gpu_id)
            total_loss += self._run_batch(input_ids, attention_mask, target_ids)
            t.set_postfix(loss= total_loss / (i+1))

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        # PATH = "checkpoint.pt"
        PATH = os.path.join(self.exp_path, f'epoch{epoch}.pth') 
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self):
        for epoch in range(self.start_epoch, self.total_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs(is_small):
    train_set = NewsDataset(is_small=is_small) 
    model =  BertLSTM1()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))

    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, exp_path:str, check_point:str, is_small:bool):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(is_small)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, exp_path, save_every, total_epochs, check_point)
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_epochs', default=200, type=int)
    parser.add_argument('--save_every', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--check_point', default=None, type=str)
    parser.add_argument('--is_small', default=False, type=bool)
    args = parser.parse_args()
    
    current_time = datetime.now()
    current_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    exp_path = os.path.join('exp', current_time)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    else:
        raise Exception('手速太快')

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, exp_path, args.check_point, args.is_small), nprocs=world_size)

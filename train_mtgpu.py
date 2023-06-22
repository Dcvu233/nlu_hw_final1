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
from model.my_model import BertLSTM1
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

def ddp_setup(rank, world_size, port='12355'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
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
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])

        if check_point == None:
            self.start_epoch = 0
        else:
            self.model.module.load_state_dict(torch.load(check_point))
            self.start_epoch = int(check_point.split('.')[0].split('epoch')[1]) + 1
        self.criterion = nn.CrossEntropyLoss()
    

    def _get_sep_pos(self, target_ids):
        sep_pos = torch.zeros((target_ids.shape[0]), dtype=torch.int32)
        for i in range(target_ids.shape[0]):
            for j in range(target_ids.shape[1]):
                if target_ids[i][j] == 102:
                    sep_pos[i] = j
                    break
        return sep_pos
    
    def _run_batch(self, input_ids, attention_mask, target_ids, with_grad=True) -> float:
        sep_pose = self._get_sep_pos(target_ids) # b
        
        if with_grad:
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, target_ids) # b,50,21128
            target = torch.zeros_like(outputs, device=outputs.device) # b,50,21128
            for i in range(outputs.shape[0]):
                for j in range(target.shape[1]):
                    target[i][j][target_ids[i][j]] = 1    
            # sep_pos = self.get_early_sep_pos(outputs, target_ids)
            loss = torch.tensor([0.0], device=outputs.device)
            for i in range(sep_pose.shape[0]):
                single = torch.tensor([0.0], device=outputs.device)
                for j in range(sep_pose[i].item()+1):
                    single += self.criterion(outputs[i][j], target[i][j])
                single /= (sep_pose[i].item()+1)
                loss += single
            

            # loss = self.criterion(output.permute(0, 2, 1), target.permute(0, 2, 1))
            # loss = self.criterion(outputs.permute(0, 2, 1)[:, :, :(sep_pose+1)], target_ids[:, :(sep_pose+1)])
            
            
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, target_ids)
                # target = torch.zeros_like(outputs)
                # for i in range(outputs.shape[0]):
                #     for j in range(target.shape[1]):
                #         target[i][j][target_ids[i][j]] = 1
                # loss = self.criterion(outputs.permute(0, 2, 1)[:, :, :(sep_pose+1)], target_ids[:, :(sep_pose+1)])
                target = torch.zeros_like(outputs, device=outputs.device) # b,50,21128
                for i in range(outputs.shape[0]):
                    for j in range(target.shape[1]):
                        target[i][j][target_ids[i][j]] = 1    
                # sep_pos = self.get_early_sep_pos(outputs, target_ids)
                loss = torch.tensor([0.0], device=outputs.device)
                for i in range(sep_pose.shape[0]):
                    single = torch.tensor([0.0], device=outputs.device)
                    for j in range(sep_pose[i].item()+1):
                        single += self.criterion(outputs[i][j], target[i][j])
                    single /= (sep_pose[i].item()+1)
                    loss += single
        return loss.item()
    
    def _run_valid(self):
        total_loss = 0.0
        for input_ids, attention_mask, target_ids in self.valid_loader:
            input_ids = input_ids.to(self.gpu_id)
            attention_mask = attention_mask.to(self.gpu_id)
            target_ids = target_ids.to(self.gpu_id)
            total_loss += self._run_batch(input_ids, attention_mask, target_ids, with_grad=False)
        return total_loss / len(self.valid_loader)
        # print(f'当前验证集上的平均loss为{total_loss / len(self.valid_loader)}')
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0
        i = 0
        tqdm_train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.total_epochs}", leave=False) 
        for input_ids, attention_mask, target_ids in tqdm_train_loader:
            # train要做6w多的batch, valid只有1000，那么每5000次train的batch之后做一轮valid
            i += 1 # 既表示训练到第几个batch（1开始计数），也表示训过的batch总数
            input_ids = input_ids.to(self.gpu_id)
            attention_mask = attention_mask.to(self.gpu_id)
            target_ids = target_ids.to(self.gpu_id)
            total_loss += self._run_batch(input_ids, attention_mask, target_ids)
            tqdm_train_loader.set_postfix(loss= total_loss / i)
            if i % 5000 == 0 or i == len(self.train_loader):
                valid_avg_loss = self._run_valid()
                if self.now_valid_loss is None or valid_avg_loss < self.now_valid_loss:
                    self.now_valid_loss = valid_avg_loss
                    if self.gpu_id == 0:
                        self._save_checkpoint(epoch, valid_avg_loss)
                print(f'当前epoch为{epoch}，已在训练集上训了{i}个batch，此时验证集上的平均loss为{valid_avg_loss}')

    def _save_checkpoint(self, epoch, valid_avg_loss):
        ckp = self.model.module.state_dict()
        ckp_path = os.path.join(self.exp_path, f'epoch{epoch}-valid_loss_{valid_avg_loss}.pth')
        torch.save(ckp, ckp_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {ckp_path}")
    # def _save_checkpoint(self, epoch):
    #     ckp = self.model.module.state_dict()
    #     ckp_path = os.path.join(self.exp_path, f'epoch{epoch}.pth') 
    #     torch.save(ckp, ckp_path)
    #     print(f"Epoch {epoch} | Training checkpoint saved at {ckp_path}")
    
    def train(self):
        self.now_valid_loss = None
        for epoch in range(self.start_epoch, self.total_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)

def load_train_objs(is_small):
    train_set = NewsDataset(is_small=is_small, mode='train')
    valid_set = NewsDataset(mode='valid') 
    model =  BertLSTM1()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))

    return train_set, model, optimizer, valid_set


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
    train_set, model, optimizer, valid_set = load_train_objs(is_small)
    train_loader = prepare_dataloader(train_set, batch_size)
    valid_loader = prepare_dataloader(valid_set, batch_size)
    trainer = Trainer(model, train_loader, valid_loader, optimizer, rank, exp_path, save_every, total_epochs, check_point)
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

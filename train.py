import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import Adam
from model.my_model import BertLSTM, BertLSTM1
from data.dataset import NewsDataset
import os
import datetime
from tqdm import tqdm
# import logging


def save_and_del(model, epoch, exp_path):
    torch.save(model.state_dict(), os.path.join(exp_path, f'epoch{epoch}.pth'))
    if epoch % 5 != 0:
        last_checkpoint_path = os.path.join(exp_path, f'epoch{epoch-1}.pth')
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)


def train(model, dataset, batch_size, epochs, exp_path, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # optimizer = Adam(model.parameters())
    # bert固定
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.CrossEntropyLoss()  # 假设0是padding的索引

    # criterion = nn.NLLLoss()
    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, target_ids in tqdm(dataloader):
            input_ids = input_ids.to(device) # b,50
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, target_ids)
            # outputs shape 是 (b, 50, 21128)
            # target_ids (b, 50)
            # outputs = outputs.view(-1, outputs.shape[-1])
            # target_ids = target_ids.view(-1)
            # target = torch.zeros((batch_size, 50, 21128)).to(device)
            target = torch.zeros_like(outputs)
            for i in range(outputs.shape[0]):
                for j in range(target.shape[1]):
                    target[i][j][target_ids[i][j]] = 1
            loss = criterion(outputs, target)
            print(loss)
            loss.backward()
            optimizer.step()
        save_and_del(model, epoch, exp_path)

def main_worker(gpu, ngpus_per_node):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345',
                            rank=gpu, world_size=ngpus_per_node)
    model = BertLSTM1()
    model = model.to(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    dataset = NewsDataset('preprocess/preprocessed_data/train_keywords.txt', 'preprocess/preprocessed_data/train_titles.txt')
    batch_size = 32
    epochs = 200
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    exp_path = os.path.join('exp', current_time)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # dataset = dataset[:32000]
    train(model, dataset, batch_size, epochs, exp_path, gpu)  

if __name__ == '__main__':


    # model = Keyword2TitleModel()
    ngpus_per_node = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

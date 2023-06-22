import torch
import torch.nn as nn
import os
import torch.distributed as dist
from data.dataset import NewsDataset
from transformers import BertTokenizer
from model.my_model import BertLSTM1

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from train_mtgpu import ddp_setup

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# # 需要
# with open("./preprocess/preprocessed_data/small_test_keywords.txt", 'r') as f:
#     while True:
#         line = f.readline()
#         if not line:
#             break
#         i = tokenizer.encode_plus(line[:-1], padding='max_length', max_length=50, return_tensors='pt')
#         testset.append(i) 
# tgt = tokenizer.encode_plus(" ", padding='max_length', max_length=50, return_tensors='pt')
# tgt = [tgt] * 4
# ddp_setup(rank, world_size)
# ddp_setup(1, 3)

# model = BertLSTM1()
# model = model.to(1)
# model = DDP(model, device_ids=[1])
# model.module.load_state_dict(torch.load('exp/2023-06-20_22:48:23/epoch17.pth'))

# with open('preprocess/preprocessed_data/small_test_keywords.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         embedding = tokenizer.encode_plus(line.strip(), padding='max_length', max_length=50, return_tensors='pt')
#         tgt = tokenizer.encode_plus(" ", padding='max_length', max_length=50, return_tensors='pt')
#         test_input = (embedding['input_ids'].to(1), embedding['attention_mask'].to(1), tgt['input_ids'].to(1))
#         softmax = nn.Softmax(dim=-1)
#         # nn.Sof
#         output = model(*test_input)
#         # print(output)
#         output = softmax(output)
#         _, output_ids = torch.max(output, dim=-1)
#         output_str = tokenizer.decode(output_ids.squeeze().tolist())
#         print(output_str)
        # sum = 0
        # for i in range(21128):
        #     sum += output[0][0][i]
        # print(f'sum={sum}')
        # print(output.shape)
        # print(output.dtype)
        # print(output)
        # print(*test_input)
        # print(output)
        # _, output_ids = torch.max(output, dim=1)
        # print(output_ids.shape)
        # print(output_ids)
# destroy_process_group()
    # for line in lines:
    #     t = tokenizer.encode_plus(line)
    #     print(type(t))
    #     # print(t.shape)
    # t = tokenizer.encode_plus(lines)
    # print(t)

def main(rank: int, world_size: int, check_point:str):
    ddp_setup(rank, world_size, port="12356")
    model = BertLSTM1()
    model1 = model.to(rank)
    model1 = DDP(model, device_ids=[rank])
    model1.module.load_state_dict(torch.load(check_point))

    with open('preprocess/preprocessed_data/small_test_keywords.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            embedding = tokenizer.encode_plus(line.strip(), padding='max_length', max_length=50, return_tensors='pt')
            tgt = tokenizer.encode_plus(" ", padding='max_length', max_length=50, return_tensors='pt')
            # test_input = (embedding['input_ids'].to(rank), embedding['attention_mask'].to(rank), tgt['input_ids'].to(rank))
            test_input = (embedding['input_ids'].to(rank), embedding['attention_mask'].to(rank), tgt['input_ids'].to(rank))
            softmax = nn.Softmax(dim=-1)
            # nn.Sof
            # output = model1(*test_input)
            output = model1(*test_input, is_train=False)
            # print(output)
            output = softmax(output)
            # print(output)
            # print(output.shape)
            # sums = torch.sum(output, dim=-1)
            # print(f'sums:{sums}')
            _, output_ids = torch.max(output, dim=-1)
            # print(output_ids)
            # print(output_ids.shape)
            output_str = tokenizer.decode(output_ids.squeeze().tolist())
            print(f'关键词：{line.strip()}')
            print(f'生成的标题：{output_str}')
    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    check_point = 'exp/2023-06-21_16:59:33/epoch18-valid_loss_0.00017228356273239363.pth'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    mp.spawn(main, args=(world_size, check_point, ), nprocs=world_size)
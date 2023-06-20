import torch
import torch.nn as nn
from model.my_model import BertLSTM
import os
import torch.distributed as dist
from data.dataset import NewsDataset, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
testset = []
with open("./preprocess/preprocessed_data/small_test_keywords.txt", 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        i = tokenizer.encode_plus(line[:-1], padding='max_length', max_length=50, return_tensors='pt')
        testset.append(i) 
tgt = tokenizer.encode_plus(" ", padding='max_length', max_length=50, return_tensors='pt')
tgt = [tgt] * 4



# def get_last_checkpoint(dir_path, index):
#     if not os.path.exists(dir_path):
#         print("dir not exists")
#     sub_dir_names = os.listdir(dir_path)
#     sub_dir_names.sort()
#     try:
#         sub_dir_name = sub_dir_names[index]
#     except Exception as e:
#         print(e)
#     sub_dir = os.path.join(dir_path, )
#     pt_files = os.listdir(sub_dir)
#     if len(pt_files) == 0:
#         index -= 1
#     for i in pt_files:

#     return os.path.join(dir_path, sub_dir_names[-1])

model = BertLSTM().cuda()
dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
model = nn.parallel.DistributedDataParallel(model, device_ids=[0])
model.load_state_dict(torch.load("exp/2023-06-19_11:55:29/epoch0.pth"))

for input, t in zip(testset, tgt):
    print(input['input_ids'], input['attention_mask'], t['input_ids'])
    y_h = model(input['input_ids'], input['attention_mask'], t['input_ids'])
    print(tokenizer.decode(y_h[0]))
# y_h = model()


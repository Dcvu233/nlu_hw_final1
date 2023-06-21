import random
import sys
sys.path.append('/home/zry/code/NLU_HW_FINAL')
import torch
from torch import nn
from transformers import BertModel
from data.dataset import NewsDataset, BertTokenizer

class BertLSTM1(nn.Module):
    def __init__(self, bert_model='bert-base-chinese', vocab_size=21128, lstm_layers=1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        # self.bert_hidden_size = self.bert.config.hidden_size
        for param in self.bert.parameters():
            param.requires_grad = False
        self.embedding = self.bert.get_input_embeddings()
        # self.lstm = nn.LSTM(input_size=self.bert_hidden_size, hidden_size=768, num_layers=lstm_layers, batch_first=True)
        self.lstm_cell = nn.LSTMCell(input_size=768, hidden_size=768)
        self.fc = nn.Linear(768, vocab_size) 

    def forward(self, keyword_ids, attention_mask, real_title_ids, is_train=True):
        b_words_code = self.bert(keyword_ids, attention_mask)[0] # torch.Size([b, 50, 768])
        #h0 c0需要的是b, 768
        # h0, c0 = encoder_output[:, -1, :].unsqueeze(0).contiguous(), encoder_output[:, -1, :].unsqueeze(0).contiguous() # torch.Size([b, 1, 768])
        h = torch.sum(b_words_code, dim=1) # b, 768
        c = h.clone()
        real_title_embedding = self.embedding(real_title_ids) # torch.Size([b, 50, 768]) 如果在训练，就用这个
        
        # outputs, _ = self.lstm(real_title_ids, (h0, c0))
        outputs = []
        for i in range(50):
            if i > 0:
                # x = outputs[-1] 
                x = real_title_embedding[:, i, :].squeeze() if is_train else outputs[-1]
            else:
                x = real_title_embedding[:, i, :].squeeze()
            h, c = self.lstm_cell(x, (h, c))
            outputs.append(h)
        # outputs 50, b, 768
        # outputs = torch.tensor(outputs).permute(1,0,2) # b, 50, 768
        outputs = torch.stack(outputs).permute(1,0,2) # b, 50, 768
        outputs = self.fc(outputs) 
        return outputs # b, 50, 21128

    
if __name__ == '__main__':   

    
    from data.dataset import NewsDataset
    test_dataset = NewsDataset()
    model = BertLSTM1()
    # print(test_dataset[0])
    y = model(test_dataset[0][0].unsqueeze(0), test_dataset[0][1].unsqueeze(0), test_dataset[0][2].unsqueeze(0))
    print(y.shape)
    res = torch.max(y, dim=2)
    # print(res.indices.shape)
    # print(res.indices)
    
    # print(y)
    # NewsDataset() 
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # print(tokenizer.decode(y[0]))
    # print(type(y))
    # for e in y:
    #     print(e.shape)
    # # print(model(x).shape)
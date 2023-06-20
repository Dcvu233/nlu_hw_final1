import sys
sys.path.append('/home/zry/code/NLU_HW_FINAL')
import torch
from torch import nn
from transformers import BertModel
from data.dataset import NewsDataset, BertTokenizer


class BertLSTM(nn.Module):
    def __init__(self, bert_model='bert-base-chinese', hidden_size=768, vocab_size=21128, lstm_layers=1):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.embedding = self.bert.get_input_embeddings()
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=768, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_output = self.bert(input_ids, attention_mask)[0]
        initial_hidden = (encoder_output[:, -1, :].unsqueeze(0).contiguous(), encoder_output[:, -1, :].unsqueeze(0).contiguous())
        decoder_input_ids = self.embedding(decoder_input_ids)
        output, _ = self.lstm(decoder_input_ids, initial_hidden)
        output = self.fc(output)
        return output

class BertLSTM1(nn.Module):
    def __init__(self, bert_model='bert-base-chinese', vocab_size=21128, lstm_layers=1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert_hidden_size = self.bert.config.hidden_size
        for param in self.bert.parameters():
            param.requires_grad = False
        self.embedding = self.bert.get_input_embeddings()
        self.lstm = nn.LSTM(input_size=self.bert_hidden_size, hidden_size=768, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(768, vocab_size)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_output = self.bert(input_ids, attention_mask)[0]
        initial_hidden = (encoder_output[:, -1, :].unsqueeze(0).contiguous(), encoder_output[:, -1, :].unsqueeze(0).contiguous())
        decoder_input_ids = self.embedding(decoder_input_ids)
        output, _ = self.lstm(decoder_input_ids, initial_hidden)
        output = self.fc(output)

        # return self.softmax(output)
        return output

class TestFC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 32)
    
    def forward(self, x):
        return self.fc(x)
    
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
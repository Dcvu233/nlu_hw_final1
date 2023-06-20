from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

class NewsDataset(Dataset):
    def __init__(self, is_small=False):
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        if is_small:
            filename = 'valid'
        else:
            filename = 'train'
        keyword_file = f'preprocess/preprocessed_data/{filename}_keywords.txt'
        title_file = f'preprocess/preprocessed_data/{filename}_titles.txt'
        with open(keyword_file, 'r', encoding='utf-8') as f:
            self.keyword_data = f.read().strip().split('\n')
        with open(title_file, 'r', encoding='utf-8') as f:
            self.title_data = f.read().strip().split('\n')

    def __len__(self):
        return len(self.keyword_data)

    def __getitem__(self, idx):
        keyword = self.keyword_data[idx]
        title = self.title_data[idx]
        inputs = self.tokenizer.encode_plus(keyword, add_special_tokens=True, padding='max_length', max_length=50, return_tensors='pt')
        outputs = self.tokenizer.encode_plus(title, add_special_tokens=True, padding='max_length', max_length=50, return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), outputs['input_ids'].squeeze()

if __name__ == '__main__':
    test_dataset = NewsDataset('preprocess/preprocessed_data/valid_keywords.txt', 'preprocess/preprocessed_data/valid_titles.txt')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
    for e in test_dataloader:
        # print(type(e))
        print(e[0].shape)
        print(e[1].shape)
        print(e[2].shape)
        assert(False)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
vocab_size = len(tokenizer.get_vocab())
print(vocab_size)  # 输出：21128

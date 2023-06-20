import json
from jieba.analyse import textrank

def get_keywords_title(news):
    # content = news['content']
    title = news['title']
    keywords = textrank(title, topK=5)
    return ' '.join(keywords) + '\n', title + '\n'

def generate_keywords_and_titles(input_file):
    with open(input_file, 'r', encoding='utf-8') as news_file:
        lines = news_file.readlines()
        print(f'总共{len(lines)}条新闻')

    filetype = input_file.split('_')[1].split('.')[0]

    if filetype == 'train':
        with open('preprocess/preprocessed_data/train_keywords.txt', 'w', encoding='utf-8') as f_keywords, open('preprocess/preprocessed_data/train_titles.txt', 'w', encoding='utf-8') as f_titles:
            for i, line in enumerate(lines):
                news = json.loads(line)
                keywords, title = get_keywords_title(news)
                if keywords == '\n' or len(title) > 48:
                    continue
                f_keywords.write(keywords)
                f_titles.write(title)
                if i % 100 == 0:
                    print(f'处理完{i}条新闻')
            
    elif filetype == 'valid':
        with open('preprocess/preprocessed_data/valid_keywords.txt', 'w', encoding='utf-8') as f_valid_keywords, open('preprocess/preprocessed_data/test_keywords.txt', 'w', encoding='utf-8') as f_test_keywords, \
            open('preprocess/preprocessed_data/valid_titles.txt', 'w', encoding='utf-8') as f_valid_titles, open('preprocess/preprocessed_data/test_titles.txt', 'w', encoding='utf-8') as f_test_titles:
            for i, line in enumerate(lines):
                news = json.loads(line)
                keywords, title = get_keywords_title(news)
                if keywords == '\n' or len(title) > 48:
                    continue
                if i % 2 == 0:
                    f_valid_keywords.write(keywords)
                    f_valid_titles.write(title)
                else:
                    f_test_keywords.write(keywords)
                    f_test_titles.write(title)
                if i % 100 == 0:
                    print(f'处理完{i}条新闻')   

generate_keywords_and_titles('preprocess/news2016zh_valid.json')
generate_keywords_and_titles('preprocess/news2016zh_train.json')
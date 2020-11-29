# pip install malaya
# constituency data preprocessing - https://github.com/huseinzol05/Malay-Dataset/blob/master/parsing/constituency/augmented.ipynb

import re
import wget
import malaya

from tqdm import tqdm
from sklearn.model_selection import train_test_split

# functions for data augmentation using malay synonyms
def case_of(text):
    return (
        str.upper
        if text.isupper()
        else str.lower
        if text.islower()
        else str.title
        if text.istitle()
        else str
    )

def remove(string):
    string = string.encode('ascii', errors='ignore').strip().decode()
    string = re.sub(r'[ ]+', ' ', string).strip()
    return string

def augment(string, n = 7):
    # look for synonyms from dictionary
    results = malaya.augmentation.synonym(string)
#     try:
#         results.extend(malaya.augmentation.transformer(string, electra))
#     except:
#         pass
    try:
        # extend possible synonyms using word embeddings
        results.extend(malaya.augmentation.wordvector(
            string, word_vector_news, soft = True
        ))
    except:
        pass
    results = list(set(results))
    results = [remove(s) for s in results if s.lower() != string.lower()]
    words = string.split()
    results = [s.split() for s in results if len([w for w in s.split() if len(w) > 1]) == len(words)]
    for i in range(len(results)):
        for no, w in enumerate(words):
            results[i][no] = case_of(w)(results[i][no])
        results[i] = ' '.join(results[i])
    return results[:n]

def replace(string, actual_string, aug_string):
    actual_string = actual_string.split()
    aug_string = aug_string.split()
    for no, word in enumerate(actual_string):
        string = string.replace(word, aug_string[no])
    return string

def data_augment(texts, parsings):
    results, out = [], []
    for i in tqdm(range(len(texts))):
        try:
            rows = augment(texts[i])
            for row in rows:
                results.append(replace(parsings[i], texts[i], row))
                out.append(row)
        except:
            pass
        results.append(parsings[i])
        out.append(texts[i])
    
    return results, out

if __name__ == '__main__':

    # get malay word embeddings for augmentation purpose
    vocab_news, embedded_news = malaya.wordvector.load_news()
    embedded_news.shape
    word_vector_news = malaya.wordvector.load(embedded_news, vocab_news)
    word_vector_news._embed_matrix

    # download indon constituency data
    url_indon_text = 'https://raw.githubusercontent.com/huseinzol05/Malay-Dataset/master/parsing/constituency/indon.txt'
    url_texts = 'https://raw.githubusercontent.com/huseinzol05/Malay-Dataset/master/parsing/constituency/texts.txt'
    wget.download(url_indon_text, out='data/texts.txt')
    wget.download(url_texts, out='data/indon.txt')
    with open('data/texts.txt') as fopen:
        texts = fopen.read().split('\n')
    with open('data/indon.txt') as fopen:
        parsing = fopen.read().split('\n')

    print(f'No. of text: {len(texts)}, No. of parsings: {len(parsing)}')

    # train test val split
    train_texts, test_val_texts, train_parsing, test_val_parsing = train_test_split(texts, parsing, test_size = 0.2, random_state=100)
    val_texts, test_texts, val_parsing, test_parsing = train_test_split(test_val_texts, test_val_parsing, test_size = 0.5, random_state=100)

    # augment data with malay synonyms
    test_aug_parsings, test_aug_texts = data_augment(test_texts, test_parsing)
    val_aug_parsings, val_aug_texts = data_augment(val_texts, val_parsing)
    train_aug_parsings, train_aug_texts = data_augment(train_texts, train_parsing)

    print(f'Augmented test data: {len(test_aug_parsings)}')
    print(f'Augmented test data: {len(val_aug_parsings)}')
    print(f'Augmented train data: {len(train_aug_parsings)}')

    # save augmented data
    with open('data/test-aug.txt', 'w') as fopen:
        fopen.write('\n'.join(test_aug_parsings))
    with open('data/test-aug-texts.txt', 'w') as fopen:
        fopen.write('\n'.join(test_aug_texts))
    with open('data/val-aug.txt', 'w') as fopen:
        fopen.write('\n'.join(val_aug_parsings))
    with open('data/val-aug-texts.txt', 'w') as fopen:
        fopen.write('\n'.join(val_aug_texts))
    with open('data/train-aug.txt', 'w') as fopen:
        fopen.write('\n'.join(train_aug_parsings))
    with open('data/train-aug-texts.txt', 'w') as fopen:
        fopen.write('\n'.join(train_aug_texts))

############## for small sample
    # with open('texts.txt') as fopen:
    #     texts = fopen.read().split('\n')
    # with open('indon.txt') as fopen:
    #     parsing = fopen.read().split('\n')

    # print(f'No. of text: {len(texts)}, No. of parsings: {len(parsing)}')

    # # train test split
    # train_texts, test_texts, train_parsing, test_parsing = train_test_split(texts[:100], parsing[:100], test_size = 0.1, random_state=100)

    # # augment data with malay synonyms
    # test_aug_parsings, test_aug_texts = data_augment(test_texts, test_parsing)
    # train_aug_parsings, train_aug_texts = data_augment(train_texts, train_parsing)

    # print(f'Augmented test data: {len(test_aug_parsings)}')
    # print(f'Augmented train data: {len(train_aug_parsings)}')

    # # save augmented data
    # with open('stest-aug.txt', 'w') as fopen:
    #     fopen.write('\n'.join(test_aug_parsings))
    # with open('stest-aug-texts.txt', 'w') as fopen:
    #     fopen.write('\n'.join(test_aug_texts))
    # with open('strain-aug.txt', 'w') as fopen:
    #     fopen.write('\n'.join(train_aug_parsings))
    # with open('strain-aug-texts.txt', 'w') as fopen:
    #     fopen.write('\n'.join(train_aug_texts))
from gensim import utils
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus = pd.read_csv("00_train_tokenized_only.csv", names=['blog', 'class'], skip_blank_lines=True)
        for line in corpus['blog']:
            # assume there's one document per line, tokens separated by whitespace
            word_tokens = line.split()
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            filtered_sentence = tokenizer.tokenize(" ".join(filtered_sentence))
            yield utils.simple_preprocess(" ".join(filtered_sentence))
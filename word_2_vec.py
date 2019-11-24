import logging 
import gensim.downloader as api
import gensim.models
from MyCorpus import MyCorpus
import time

start = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, workers=4)
end = time.time()
print("train_time ", end-start)
model.wv.save_word2vec_format('word_2_vec_normalized.txt')
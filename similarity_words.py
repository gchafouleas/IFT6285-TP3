import spacy
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

nlp = spacy.load('word_2_vec_normalized_size_200_min_count_10')
vocab = list(nlp.vocab.strings)
print(len(vocab))
vocab.remove('nt')
vocab = [v for v in vocab if re.match('^[a-zA-z]+$', v) is not None]
print(vocab[0])
vocab = [token for token in vocab if not token in stop_words]
vocab = [token for token in vocab if token not in string.punctuation]
print(len(vocab))
tokens = nlp("alchool beach xenphobic cold hot because the person want christmas construction narita boolean to be chris hello person nuts tasty perpetuates craving clinic showers inarticulate autumn classics australia")
#tokens = nlp("nuts tasty perpetuates craving clinic showers inarticulate autumn classics australia")

#tokens = vocab[:10]

similar_words_text = []
len_tokens = len(tokens)
count = 1
for token1 in tokens:
    print("Item : {0}/{1}".format(count, len_tokens))
    word = token1
    similar_words = []
    similar_word_entry = []
    similar_words_similarity_value = []
    count_similar_words = 0
    for v in vocab:
        token2 = nlp(v)
        print(token2)
        if token2.text != token1.text and token1.similarity(token2) > 0.5:
            similar_words.append(token2.text)
            similar_words_similarity_value.append(token1.similarity(token2))
            count_similar_words +=1
    if count_similar_words != 0:
        index_words = np.array(similar_words_similarity_value).argsort()[::-1]
        similar_word_entry.append(word.text)
        similar_word_entry.append(count_similar_words)
        similar_words = np.array(similar_words)[index_words]
        if len(similar_words) > 13:
            similar_words = similar_words[:20]
        similar_word_entry.append(','.join(similar_words))
        similar_words_text.append(similar_word_entry)
    count +=1 

similar_words_text =  np.array(similar_words_text).reshape(-1,3)
similar_words_text = similar_words_text[similar_words_text[:, 1].argsort()]
np.savetxt('similarity_size_200_min_count_10.out', similar_words_text, delimiter='   ', fmt="%s")
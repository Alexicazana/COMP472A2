# Evaluation of the word2vec-google-news-300 Pre-trained Model
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np

model = Word2Vec.load("word2vec-google-news-300")  # load word2vec-google-news-300 embedding model
model.wv.similarity("question_word")  # show most similar synonym for the question word; gotta change word later


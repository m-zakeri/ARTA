from copy import deepcopy as dc

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec


def preprocess(corpus_path):
    lem = WordNetLemmatizer()
    lemmed = []
    cleaned_docs = []

    # Preprocess each document
    for i in range(10000):
        lemmed.clear()
        doc = open(corpus_path + str(i + 1) + ".txt", 'r', encoding='utf-8')
        # convert to lower case
        lines_doc = doc.read().lower()
        doc.close()

        # tokenize document
        tokens = word_tokenize(lines_doc)
        # remove stopwords
        tokens_without_sw = [word for word in tokens if not word in stopwords]

        # lemmatize words
        for j in tokens_without_sw:
            l = lem.lemmatize(j)
            if len(l) > 2:
                lemmed.append(l)
        cleaned_docs.insert(i, dc(lemmed))


def compute_vector_similarity(sentences_list):
    model = Word2Vec(sentences_list, window=10, min_count=5, vector_size=50, sg=1)

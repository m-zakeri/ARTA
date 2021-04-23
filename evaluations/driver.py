"""

"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from gensim.models import Word2Vec

# nltk.download('stopwords')
all_stopwords = stopwords.words('english')

additional_stop_word_list1 = ['===', '====', '...', 'one', 'e.g', 'two', 'three', 'four', 'five', 'six', 'seven',
                              'eight', 'nine', 'ten', 'e.g.']
additional_stop_word_list2 = ['|', '=', '&', '{', '}', '[', ']', '>', '<', '?', '!', '$', '%', '#', "'", '--', ')', '(',
                              "''", '``', ':', ';', "'s", '10', '6', '7', '8', '9', '5', '4', '3', '0', '1', '2', 'j.',
                              'c.', 'm.', 'a.', '\\', '^', 'x', 'h', 'q', 'l', 'w', 'g', 'c', 'n', 'f', 'r', 'k', 'p',
                              'j', 'e', 'b', 'u', 'v', 'le', 'de', ',', '.', '==', '+', '–', '-', '—', '−', '_']

for n in additional_stop_word_list1:
    all_stopwords.append(n)

print(all_stopwords)
print(len(all_stopwords))

tokens = word_tokenize(
    'When the train is stationary or after a certain time (e.g. the time for "route releasing" of the overlap, the release speed calculation shall be based on the distance to the danger point (if calculated on-board). The condition for this change shall be defined for each target as infrastructure data.')

lem = WordNetLemmatizer()
for j in tokens:
    l = lem.lemmatize(j)
    print(l)


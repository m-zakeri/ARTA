import re
from abc import ABC, abstractmethod
from functools import lru_cache

from textblob import TextBlob, Word
from textblob.exceptions import MissingCorpusError

from .models import Finding, Smell


@lru_cache(maxsize=None)
def get_smell_object(title):
    return Smell.objects.get_or_create(title=title)[0]


@lru_cache(maxsize=None)
def get_dictionary(smell_title):
    smell_object = get_smell_object(smell_title)
    return smell_object.dictionary.all().values()


class AbstractDetector(ABC):

    @property
    def message(self):
        raise NotImplemented

    @property
    def title(self):
        raise NotImplemented

    def extract_words(self, requirement):
        """Returns an list of tuples of the form (word, POS tag).
            Example:
            [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'),('Thursday', 'NNP'), ('morning', 'NN')]
        """
        return TextBlob(requirement.description).tags

    @abstractmethod
    def is_smell(self, word):
        raise NotImplemented

    def get_message(self) -> str:
        return f'{self.message}\n'

    def check_multiple_words(self, requirement):
        return

    def find(self, requirement):
        temp_req = requirement.description
        for word in self.extract_words(requirement):
            index = temp_req.index(word[0])
            temp_req = temp_req.replace(word[0], '*' * len(word[0]), 1)
            word_object = self.is_smell(word)
            if word_object:
                self.create_object(index, requirement, word[0], word_object)
        self.check_multiple_words(requirement)

    def create_object(self, index, requirement, word, word_object=None):
        Finding.objects.get_or_create(
            smell=get_smell_object(self.title),
            index_start=index,
            index_stop=index + len(word),
            requirement=requirement,
        )


class AbstractDictionaryDetector(AbstractDetector):
    def is_smell(self, word):
        word, tag = re.sub(r'[^\w]', '', word[0].lower()), word[1]
        blob_word = Word(word)
        dictionary = get_dictionary(self.title)
        flag = dictionary.filter(word=word).first()
        if flag:
            return flag
        try:
            flag = dictionary.filter(word=blob_word.lemmatize(tag)).first()
        except MissingCorpusError:
            return False
        return flag

    def create_object(self, index, requirement, word, word_object=None):
        Finding.objects.get_or_create(
            smell=get_smell_object(self.title),
            index_start=index,
            index_stop=index + len(word),
            requirement=requirement,
            word_dictionary_id=word_object['id']
        )

    def check_multiple_words(self, requirement):
        description = requirement.description.lower()
        description_blob = TextBlob(description)
        for word in get_dictionary(self.title):
            ngrams = description_blob.ngrams(len(word['word'].split()))
            for ngram in ngrams:
                ngram = ' '.join(ngram)
                word_object = get_dictionary(self.title).filter(word=ngram).first()
                if word_object:
                    index = description.find(ngram)
                    if index != -1:
                        self.create_object(index, requirement, ngram, word_object)


class AbstractTaggingDetector(AbstractDetector):
    def extract_words(self, requirement):
        blob = TextBlob(requirement.description)
        return blob.tags

    def is_smell(self, word):
        raise NotImplemented

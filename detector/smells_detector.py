from textblob import TextBlob

from .abstract_detector import AbstractTaggingDetector, AbstractDictionaryDetector, get_smell_object
from .models import Finding


class SubjectiveLanguageSmell(AbstractDictionaryDetector):
    title = 'Subjective Language'
    message = 'Subjective Language refers to words of which ' \
              'the semantics is not objectively defined'


class AmbiguousAdverbAdjectiveSmell(AbstractDictionaryDetector):
    title = 'Ambiguous Adverbs and Adjectives'
    message = 'Ambiguous Adverbs and Adjectives ' \
              'refer to certain adverbs and adjectives that are unspecific by nature'


class LoopholesSmell(AbstractDictionaryDetector):
    title = 'Loopholes'
    message = 'Loopholes refer to phrases that express that the following requirement' \
              'must be fulfilled only to a certain, imprecisely defined extent.'


class OpenEndedTermsSmell(AbstractDictionaryDetector):
    title = 'Open-ended, non-verifiable terms'
    message = 'Open-ended terms are hard to verify as they ' \
              'offer a choice of possibilities for the developers.'


class SuperlativeSmell(AbstractTaggingDetector):
    title = 'Superlatives'
    message = 'Superlatives refer to requirements that express a ' \
              'relation of the system to all other systems'

    def is_smell(self, word):
        return word[1] == 'JJS' or word[1] == 'RBS'


class ComparativesSmell(AbstractTaggingDetector):
    title = 'Comparatives'
    message = 'Comparatives are used in requirements that ' \
              'express a relation of the system to specific other systems or previous situations.'

    def is_smell(self, word):
        return word[1] == 'RBR' or word[1] == 'JJR'


class VaguePronounSmell(AbstractTaggingDetector):
    title = 'Vague Pronoun'
    message = 'Vague Pronouns are unclear relations of a pronoun'

    def find(self, requirement):
        blob = TextBlob(requirement.description)
        pos_tags = blob.tags
        tags = list(map(lambda x: x[1], pos_tags))
        start = 0
        temp_req = requirement.description
        for i, word in enumerate(pos_tags):
            word, tag = word
            index = temp_req.index(word)
            if tag == 'WDT':
                tag_index = tags.index('WDT')
                tags[tag_index] = 'reviewed'
                sub_tags = tags[start:tag_index]
                count = sub_tags.count('NN') + sub_tags.count('NNS') + sub_tags.count('NNP') + sub_tags.count('NNPS')
                if count > 1:
                    word_index = temp_req.index(word)
                    Finding.objects.create(
                        smell=get_smell_object(self.title, self.message),
                        index_start=word_index,
                        index_stop=word_index + len(word),
                        requirement=requirement,
                    )
                start = tag_index + 1
            temp_req = temp_req[0:index] + '*' * len(word) + temp_req[index + len(word)::]


class NegativeStatementSmell(AbstractTaggingDetector):
    title = 'Negative Statement'
    message = 'Negative Statements are statements of system capability not to be provided'

    def is_smell(self, word):
        return word[1] == 'RB' and (word[0].lower() == 'no' or word[0].lower() == 'not')

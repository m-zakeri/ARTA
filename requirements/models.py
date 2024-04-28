import inspect

from django.db import models
from django.utils.translation import ugettext_lazy as _
from textblob import TextBlob

from detector import smells_detector
from projects.models import Project


class Requirement(models.Model):
    TYPE_FUNCTIONAL = 'F'
    TYPE_NONFUNCTIONAL = 'NF'

    TYPE_CHOICES = [
        (TYPE_FUNCTIONAL, _('functional')),
        (TYPE_NONFUNCTIONAL, _('non-functional'))
    ]

    title = models.CharField(max_length=255, verbose_name=_('title'))
    number = models.CharField(max_length=16, verbose_name=_('requirement id'))
    description = models.TextField(
        verbose_name=_('description'), help_text=_('actual requirement'),
    )

    requirement_type = models.CharField(max_length=2, choices=TYPE_CHOICES, verbose_name=_('type'),
                                        default=TYPE_FUNCTIONAL)

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='requirements', verbose_name=_('project')
    )

    created_date = models.DateTimeField(auto_now_add=True, verbose_name=_('created date'))
    modified_date = models.DateTimeField(auto_now=True, verbose_name=_('modified date'))

    score = models.FloatField(default=0.0, verbose_name=_('score'))
    cleanness = models.FloatField(default=0.0, verbose_name=_('cleanness'))
    testability = models.FloatField(default=0.0, verbose_name=_('testability'))
    polarity = models.FloatField(default=0.0, verbose_name=_('polarity'))
    subjectivity = models.FloatField(default=0.0, verbose_name=_('subjectivity'))
    readability = models.FloatField(default=0.0, verbose_name=_('readability'))

    class Meta:
        verbose_name = _('requirement')
        verbose_name_plural = _('requirements')

    def __str__(self):
        return self.number

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        sentiment = self.calculate_sentiment()
        self.score = self.calculate_scores()
        self.cleanness = self.calculate_cleanness()
        self.testability = self.calculate_testability()
        self.polarity = sentiment.polarity
        self.subjectivity = sentiment.subjectivity * 100
        return super(Requirement, self).save(force_insert, force_update, using, update_fields)

    def find_smells(self):
        self.smells.filter().delete()
        for name, obj in inspect.getmembers(smells_detector):
            if inspect.isclass(obj) and obj.__name__[-5:] == 'Smell':
                detector = obj()
                detector.find(self)

    def calculate_cleanness(self):
        """
        https://tutorial.djangogirls.org/en/django_orm/
        :return:
        """
        with_smell_words = 0
        smell_types = set()
        reviewed = set()
        for finding in self.smells.all():
            location = (finding.index_start, finding.index_stop)
            smell_types.add(finding.smell.title)
            # if location in reviewed:
            #     continue
            reviewed.add(location)

            word = self.description[finding.index_start:finding.index_stop]
            with_smell_words += len(word.split())

        # print('@@@@', smell_types)
        blob = TextBlob(self.description)
        requirement_length = len(blob.tokens)

        # A clean requirement
        if with_smell_words == 0 and len(smell_types) == 0:
            return 1

        # Not a clean requirement
        return 1 - (with_smell_words / requirement_length) ** (1 / len(smell_types)) #len(smell_types))

    def calculate_testability(self):
        epsilon = [0.01, 0.50, 0.99]
        cleanness = self.calculate_cleanness()
        blob = TextBlob(self.description)
        requirement_sentences_length = len(blob.sentences)
        return cleanness/((1+epsilon[1])**(requirement_sentences_length-1))

    def calculate_scores(self):
        with_smell_words = 0
        reviewed = set()
        for finding in self.smells.all():
            location = (finding.index_start, finding.index_stop)
            # if location in reviewed:
            #     continue
            reviewed.add(location)
            word = self.description[finding.index_start:finding.index_stop]
            with_smell_words += len(word.split())
        return (1 - with_smell_words / len(self.description.split())) * 100

    def calculate_sentiment(self):
        blob = TextBlob(self.description)
        return blob.sentiment

    @property
    def get_smells_count(self):
        return self.smells.count()

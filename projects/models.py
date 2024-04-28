from django.db import models
from django.utils.translation import ugettext_lazy as _


class Project(models.Model):
    name = models.CharField(max_length=255, unique=True, verbose_name=_('name'))
    description = models.TextField(blank=True, default='', verbose_name=_('description'))

    created_date = models.DateTimeField(auto_now_add=True, verbose_name=_('created date'))
    modified_date = models.DateTimeField(auto_now=True, verbose_name=_('modified date'))

    class Meta:
        verbose_name = _('project')
        verbose_name_plural = _('projects')

    def __str__(self):
        return self.name

    @property
    def score(self):
        return self.requirements.all().aggregate(models.Avg('score'))['score__avg']

    @property
    def polarity(self):
        return self.requirements.all().aggregate(models.Avg('polarity'))['polarity__avg']

    @property
    def subjectivity(self):
        return self.requirements.all().aggregate(models.Avg('subjectivity'))['subjectivity__avg']

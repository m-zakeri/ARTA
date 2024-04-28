from django.contrib.auth import get_user_model
from django.db import models
from django.utils.translation import ugettext_lazy as _


class Smell(models.Model):
    title = models.CharField(max_length=255, verbose_name=_('smell title'), unique=True)
    description = models.CharField(max_length=512, verbose_name=_('smell description'))
    tip = models.CharField(max_length=512, verbose_name=_('tip'), default='', blank=True)

    class Meta:
        verbose_name = _('smell')
        verbose_name_plural = _('smells')

    def __str__(self):
        return self.title


class Word(models.Model):
    smell = models.ForeignKey(Smell, verbose_name=_('smell'), related_name='dictionary', on_delete=models.CASCADE)
    word = models.CharField(max_length=255, verbose_name=_('word'))
    tip = models.TextField(verbose_name=_('tip'), default='', blank=True)

    class Meta:
        verbose_name = _('word')
        verbose_name_plural = _('words')
        unique_together = ('word', 'smell')

    def __str__(self):
        return f'{self.smell.title}: {self.word}'

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        self.word = self.word.lower()
        return super(Word, self).save(force_insert, force_update, using, update_fields)


class Finding(models.Model):
    smell = models.ForeignKey(Smell, verbose_name=_('smell'), on_delete=models.CASCADE, related_name='findings')

    index_start = models.IntegerField(verbose_name=_('index start at'))
    index_stop = models.IntegerField(verbose_name=_('index stop at'))

    is_ok = models.BooleanField(default=False, verbose_name=_('is ok?'))
    is_manual = models.BooleanField(default=False, verbose_name=_('is added by hand?'))
    is_reviewed = models.BooleanField(default=False, verbose_name=_('is reviewed?'))

    requirement = models.ForeignKey('requirements.Requirement', verbose_name=_('requirements'),
                                    related_name='smells', on_delete=models.CASCADE)

    word = models.CharField(max_length=128, default='', blank=True, verbose_name=_('enter word'))
    word_dictionary = models.ForeignKey(Word, verbose_name=_('dictionary member'), related_name='findings',
                                        on_delete=models.CASCADE, null=True, blank=True)

    created_date = models.DateTimeField(auto_now_add=True, verbose_name=_('created date'))
    reviewed_date = models.DateTimeField(auto_now=True, verbose_name=_('reviewed date'))

    class Meta:
        verbose_name = _('finding')
        verbose_name_plural = _('findings')

    def __str__(self):
        return f'{self.requirement}:{self.index_start}-{self.index_stop}'

    def view_smell(self):
        return f'{self.requirement.description[self.index_start:self.index_stop]} : {self.requirement.description}'

    @property
    def render_smell(self):
        description = self.requirement.description
        description = f'<p>' + \
                      description[0:self.index_start] + \
                      f'<mark style="background-color:yellow"">' + \
                      description[self.index_start:self.index_stop] + \
                      '</mark>' + description[self.index_stop::] + '</p>'
        if self.word_dictionary and self.word_dictionary.tip:
            description += '<p class="text-secondary small">hint: ' + self.word_dictionary.tip + '</p>'
        description = f'<div class="requirement_content" id="finding_{self.id}" hidden> {description} </div>'
        return description


class Comment(models.Model):
    content = models.TextField(verbose_name=_('content'))
    user = models.ForeignKey(get_user_model(), verbose_name=_('user'), on_delete=models.CASCADE,
                             related_name='comments')
    finding = models.ForeignKey(Finding, verbose_name=_('findings'), on_delete=models.CASCADE, related_name='comments')
    created_date = models.DateTimeField(auto_now_add=True, verbose_name=_('created date'))

    class Meta:
        verbose_name = _('comment')
        verbose_name_plural = _('comments')

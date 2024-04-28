from django.contrib import admin
from django.db.models import Count
from django.utils.translation import gettext_lazy as _

from detector.models import Finding
from .models import Requirement


class HaveSmellFilter(admin.SimpleListFilter):
    title = _('have smell')
    parameter_name = 'have_smell'

    def lookups(self, request, model_admin):
        return (
            ('true', _('yes')),
            ('false', _('no'))
        )

    def queryset(self, request, queryset):
        queryset = queryset.annotate(smell_count=Count('smells'))
        if self.value() == 'true':
            return queryset.filter(smell_count__gte=1)
        return queryset.filter(smell_count__exact=0)


class FindingInLine(admin.StackedInline):
    fields = ['smell', ('index_start', 'index_stop'), 'is_ok', 'requirement', 'view_smell', 'is_manual', 'is_reviewed',
              'word', 'word_dictionary']
    readonly_fields = ['created_date', 'reviewed_date', 'view_smell']
    model = Finding
    extra = 0
    show_change_link = True


@admin.register(Requirement)
class RequirementAdmin(admin.ModelAdmin):
    list_display = ('id', 'number', 'title', 'requirement_type', 'description', 'project', 'created_date',
                    'modified_date', 'score', 'polarity', 'subjectivity')
    raw_id_fields = ('project',)
    fields = ('title', 'number', 'description', 'requirement_type', 'project', 'get_smells_count', 'score', 'polarity',
              'subjectivity')
    readonly_fields = ('created_date', 'modified_date', 'get_smells_count', 'score', 'polarity', 'subjectivity')
    list_filter = ('project', 'requirement_type', HaveSmellFilter)
    ordering = ('number',)
    search_fields = ('description', 'number')
    inlines = [FindingInLine]

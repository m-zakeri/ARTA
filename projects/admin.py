from django.contrib import admin

from requirements.models import Requirement
from .models import Project


class RequirementInLine(admin.TabularInline):
    model = Requirement
    fields = ('title', 'number', 'description', 'requirement_type', 'project', 'get_smells_count')
    readonly_fields = ('smells', 'created_date', 'modified_date', 'get_smells_count')
    extra = 0
    show_change_link = 'title'


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    fields = ('name', 'description', 'score', 'polarity', 'subjectivity')
    readonly_fields = ('created_date', 'modified_date', 'score', 'polarity', 'subjectivity')
    list_display = ('id', 'name', 'created_date', 'modified_date', 'score', 'polarity', 'subjectivity')
    inlines = [RequirementInLine]

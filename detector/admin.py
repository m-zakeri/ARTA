from django.contrib import admin

from .models import Finding, Smell, Word, Comment


class WordInLine(admin.TabularInline):
    model = Word
    fields = ['smell', 'word', 'tip']
    extra = 0


class CommentInLine(admin.TabularInline):
    model = Comment
    list_display = ['content', 'user', 'related_name', 'finding', 'created_date', ]
    extra = 0


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['content', 'user', 'finding', 'created_date', ]
    fields = ['content', 'user', 'finding', 'created_date', ]
    readonly_fields = ['created_date']
    sortable_by = ['created_date']
    search_fields = ['content']
    list_filter = ['user']


@admin.register(Finding)
class FindingAdmin(admin.ModelAdmin):
    list_display = ['id', 'smell', 'requirement', 'view_smell', 'word', 'index_start', 'index_stop', 'is_ok',
                    'is_manual', 'is_reviewed']
    list_editable = ['is_ok', 'is_reviewed', 'word']
    fields = ['smell', ('index_start', 'index_stop'), 'is_ok', 'requirement', 'view_smell', 'is_manual', 'is_reviewed',
              'word', ]
    ordering = ['id']
    list_filter = ['is_ok', 'smell', 'is_manual', 'is_reviewed']
    readonly_fields = ['created_date', 'reviewed_date', 'view_smell']
    search_fields = ['requirement__number']
    inlines = [CommentInLine]


@admin.register(Smell)
class SmellAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'description']
    search_fields = ['title', 'description']
    inlines = [WordInLine]


@admin.register(Word)
class WordAdmin(admin.ModelAdmin):
    list_display = ['id', 'word', 'smell']
    search_fields = ['word', 'tip']
    list_filter = ['smell']
    fields = ['smell', 'word', 'tip']

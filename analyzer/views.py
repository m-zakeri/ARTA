from datetime import datetime
from tempfile import TemporaryFile
from threading import Lock

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import xlsxwriter
from django.db.models import Count
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.views.generic.base import TemplateView
from pandas import DataFrame

from analyzer.utils import get_projects_size, get_figure_temp_file, get_colors
from detector.models import Finding
from projects.models import Project
from requirements.models import Requirement

plot_lock = Lock()


def get_excel(request, by_expert=''):
    kwargs = {}
    if by_expert == 'true':
        kwargs['is_manual'] = True
    else:
        kwargs['is_reviewed'] = True
        kwargs['is_ok'] = False
    excel_file = TemporaryFile('w+b')
    workbook = xlsxwriter.Workbook(excel_file)
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, 'text')
    worksheet.write(0, 1, 'filename')
    worksheet.write(0, 2, 'subjective_language')
    worksheet.write(0, 3, 'ambiguous_adverbs_adjectives')
    worksheet.write(0, 4, 'loopholes')
    worksheet.write(0, 5, 'open_ended')
    worksheet.write(0, 6, 'superlatives')
    worksheet.write(0, 7, 'comparatives')
    worksheet.write(0, 8, 'negative_statements')
    worksheet.write(0, 9, 'vague_pronouns')

    row = 1

    for project in Project.objects.all():
        project_file = project.name
        for requirement in project.requirements.all():
            requirement_text = requirement.description

            subjective_language = requirement.smells.filter(smell__title='Subjective Language', **kwargs) \
                .values('index_start', 'index_stop')
            subjective_language = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']], subjective_language)))

            ambiguous_adverbs_adjectives = requirement.smells \
                .filter(smell__title='Ambiguous Adverbs and Adjectives', **kwargs) \
                .values('index_start', 'index_stop')
            ambiguous_adverbs_adjectives = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']],
                        ambiguous_adverbs_adjectives)))

            loopholes = requirement.smells.filter(smell__title='Loopholes', **kwargs) \
                .values('index_start', 'index_stop')
            loopholes = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']], loopholes)))

            open_ended = requirement.smells.filter(smell__title='Open-ended, non-verifiable terms', **kwargs) \
                .values('index_start', 'index_stop')
            open_ended = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']], open_ended)))

            superlatives = requirement.smells.filter(smell__title='Superlatives', **kwargs) \
                .values('index_start', 'index_stop')
            superlatives = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']], superlatives)))

            comparatives = requirement.smells.filter(smell__title='Comparatives', **kwargs) \
                .values('index_start', 'index_stop')
            comparatives = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']], comparatives)))

            negative_statements = requirement.smells.filter(smell__title='Negative Statement', **kwargs) \
                .values('index_start', 'index_stop')
            negative_statements = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']], negative_statements)))

            vague_pronouns = requirement.smells.filter(smell__title='Vague Pronoun', **kwargs) \
                .values('index_start', 'index_stop')
            vague_pronouns = '*'.join(
                set(map(lambda x: requirement_text[x['index_start']:x['index_stop']], vague_pronouns)))

            worksheet.write(row, 0, requirement_text)
            worksheet.write(row, 1, project_file)
            worksheet.write(row, 2, subjective_language)
            worksheet.write(row, 3, ambiguous_adverbs_adjectives)
            worksheet.write(row, 4, loopholes)
            worksheet.write(row, 5, open_ended)
            worksheet.write(row, 6, superlatives)
            worksheet.write(row, 7, comparatives)
            worksheet.write(row, 8, negative_statements)
            worksheet.write(row, 9, vague_pronouns)
            row += 1
    workbook.close()
    excel_file.seek(0)
    response = HttpResponse(excel_file.read(),
                            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = \
        f'attachment; filename=dataset_{str(datetime.now())}_' \
        f'{"expert_smells" if by_expert == "true" else "smella_smells"}.xlsx'
    return response


def tree_map_view(request, project_name=''):
    project_obj = get_object_or_404(Project, name=project_name)

    requirements = Requirement.objects \
        .annotate(smells_count=Count('smells')) \
        .filter(project=project_obj) \
        .values_list('smells_count', 'description')

    requirement_smells_count = list(map(lambda x: x[0], requirements))
    requirement_smells_count_colors, colors_dic = get_colors(requirement_smells_count)
    requirements_size = list(map(lambda x: len(x[1]), requirements))
    plot_lock.acquire()
    plt.rc('font', size=32)
    tree_map_plot = squarify.plot(
        sizes=requirements_size, color=requirement_smells_count_colors
    )
    color_table = tree_map_plot.table(
        loc='right', cellLoc='center',
        cellText=list(map(lambda x: [x], colors_dic.keys())),
        cellColours=list(map(lambda x: [x], colors_dic.values())),
        colWidths=[0.05]
    )
    color_table.scale(1, 1.5)
    plt.axis('off')
    figure_temp_file = get_figure_temp_file(color_table, width=50)
    plot_lock.release()
    return HttpResponse(figure_temp_file, content_type='image/jpg')


def requirement_size_view(request):
    project_name, project_values = get_projects_size()
    plot_lock.acquire()
    plt.rcParams["font.size"] = "10"
    box_plot = sns.boxplot(data=project_values)
    plt.xticks(plt.xticks()[0], project_name)
    figure_temp_file = get_figure_temp_file(box_plot)
    plot_lock.release()
    return HttpResponse(figure_temp_file, content_type='image/jpg')


def size_smell_rel_view(request):
    project_name, project_values = get_projects_size()
    project_values = list(map(lambda x: sum(x), project_values))
    smell_count = Project.objects.annotate(
        smell_count=Count('requirements__smells')
    ).values_list('smell_count', flat=True)
    data = list(map(lambda x: [project_name[x], project_values[x], smell_count[x]], range(len(project_name))))
    df = DataFrame(data, columns=['name', 'size in words', 'number of smells'])
    plot_lock.acquire()
    plt.rcParams["font.size"] = "32"
    scatter_plot = sns.scatterplot(x='size in words', y='number of smells', hue='name', data=df)
    figure_temp_file = get_figure_temp_file(scatter_plot)
    plot_lock.release()
    return HttpResponse(figure_temp_file, content_type='image/jpg')


class MetricView(TemplateView):
    template_name = 'analyzer/metrics.html'

    def get_context_data(self, **kwargs):
        """
        detecting smell is positive
        no smell is negative
        """
        context = super().get_context_data(**kwargs)
        detected_findings = Finding.objects.filter(is_reviewed=True, is_manual=False)
        total_findings = detected_findings.count()
        if not total_findings:
            context['msg'] = 'there is no reviewed finding, review them from the admin page first'
            return context

        # we classified a finding as false positive if the finding was
        # not an instance of the smell
        false_positive = detected_findings.filter(is_ok=True, is_manual=False).count()
        true_positive = detected_findings.filter(is_ok=False, is_manual=False).count()
        false_negative = Finding.objects.filter(is_manual=True).count()

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        context['total_findings'] = total_findings
        context['false_positive'] = false_positive
        context['false_negative'] = false_negative
        context['true_positive'] = true_positive
        context['precision'] = precision
        context['recall'] = recall
        context['projects'] = Project.objects.all().values_list('name', flat=True)

        smells_precision = {}
        smells_recall = {}
        smells_data = {}

        smells_false_positive = detected_findings.filter(is_ok=True, is_manual=False).values('smell__title') \
            .annotate(count=Count('smell'))
        smells_true_positive = detected_findings.filter(is_ok=False, is_manual=False).values('smell__title') \
            .annotate(count=Count('smell'))
        smells_false_negative = Finding.objects.filter(is_manual=True).values('smell__title') \
            .annotate(count=Count('smell'))

        for tp in smells_true_positive:
            smells_data[tp['smell__title']] = {'precision': tp['count'], 'recall': tp['count'],
                                               'title': tp['smell__title']}

        for fp in smells_false_positive:
            smells_data[fp['smell__title']]['precision'] /= smells_data[fp['smell__title']]['precision'] + fp['count']

        for fn in smells_false_negative:
            smells_data[fn['smell__title']]['recall'] /= smells_data[fn['smell__title']]['recall'] + fn['count']

        context['smells'] = smells_data

        return context

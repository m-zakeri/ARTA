import random
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.backends.backend_agg import FigureCanvasAgg

from projects.models import Project


def get_projects_size():
    project_name = tuple()
    project_values = tuple()
    for project in Project.objects.all():
        descriptions_count = project.requirements.all().values_list('description', flat=True)
        descriptions_count = list(map(lambda x: len(x), descriptions_count))
        project_name += (project.name,)
        project_values += (descriptions_count,)
    return project_name, project_values


def get_figure_temp_file(box_plot, width=25, height=25):
    plt.close()
    figure = box_plot.get_figure()
    figure.set_size_inches(width, height)
    canvas = FigureCanvasAgg(figure)
    fp = TemporaryFile()
    canvas.print_jpg(fp)
    fp.seek(0)
    data = fp.read()
    fp.close()
    return data


def get_colors(items):
    result = {}
    for i in set(items):
        random_color = random.randint(1048576, 16777215)
        hex_color = str(hex(random_color))
        result[i] = '#' + hex_color[2:]
    return list(map(lambda x: result.get(x), items)), result

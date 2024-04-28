from django import template
from dominate.tags import *

from detector import run_detection

register = template.Library()


@register.simple_tag(takes_context=True)
def render_smells(context):
    requirement = context['requirement']
    smells = run_detection(requirement.description)
    result = p(_class='lead')
    founded_smells = set()
    for index, word in enumerate(requirement.description.split()):
        smell = smells.get(index)
        if smell:
            smell_title, smell = smell.split(':')
            smell_tag = u(word, _class='smell', data_toggle='popover', title=smell_title, data_content=smell,
                          data_placement='bottom')
            founded_smells.add(smell_title)
            result.appendChild(smell_tag)
        else:
            result.appendChild(word + ' ')
    context['founded_smells'] = founded_smells
    return result

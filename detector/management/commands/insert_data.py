import json

from django.core.management.base import BaseCommand

from projects.models import Project
from requirements.models import Requirement


class Command(BaseCommand):
    help = 'Closes the specified poll for voting'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str)
        parser.add_argument('project_name', type=str)
        parser.add_argument('--title', action='store_true')

    def handle(self, *args, **options):
        f = open(options['file_path'], 'r')
        requirements_json = json.load(f)
        f.close()
        project, is_created = Project.objects.get_or_create(name=options['project_name'])
        if is_created:
            self.stdout.write(f'project {project} created')

        for requirement in requirements_json:
            if options['title']:
                title = f'requirement {requirement.get("requirement id")}'
            else:
                title = requirement.get('title')
            new_requirement, is_req_created = Requirement.objects.get_or_create(
                project=project,
                title=title,
                description=' '.join(requirement.get('description').split()),
                number=requirement.get('requirement id'),
                requirement_type=Requirement.TYPE_FUNCTIONAL if requirement.get('type').lower() == 'functional'
                else Requirement.TYPE_NONFUNCTIONAL
            )

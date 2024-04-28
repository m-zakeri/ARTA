import re

from django.core.management.base import BaseCommand
from django.urls import reverse

from requirements.models import Requirement


class Command(BaseCommand):
    def handle(self, *args, **options):
        requirements = set()
        id_map = {}

        for requirement in Requirement.objects.all():
            req_id = requirement.id
            requirement_text = re.sub(r'[^\w]', ' ', requirement.description.lower())
            requirement_text = ' '.join(requirement_text.split())

            if requirement_text in requirements:
                print(req_id)
                print(requirement.project)
                print(requirement.description)
                print(requirement_text)
                print(id_map[requirement_text])

                print('http://127.0.0.1:8000' + reverse(
                    'admin:%s_%s_change' % (requirement._meta.app_label, requirement._meta.model_name),
                    args=[req_id]))

                print('http://127.0.0.1:8000' + reverse(
                    'admin:%s_%s_change' % (requirement._meta.app_label, requirement._meta.model_name),
                    args=[id_map[requirement_text]]))
                requirement.delete()

            else:
                requirements.add(requirement_text)
                id_map[requirement_text] = req_id

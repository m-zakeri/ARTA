from django.core.management.base import BaseCommand

from requirements.models import Requirement


class Command(BaseCommand):
    def handle(self, *args, **options):
        for requirement in Requirement.objects.all():
            requirement.find_smells()

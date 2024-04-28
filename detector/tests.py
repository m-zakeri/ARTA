from django.test import TestCase

from requirements.models import Requirement
from projects.models import Project
from .models import Smell
from . import run_detection
from .smells_detector import SubjectiveLanguageSmell, SuperlativeSmell


class SubjectiveTest(TestCase):
    def setUp(self) -> None:
        self.project = Project.objects.create(name='test project')
        self.requirement = Requirement.objects.create(
            project=self.project,
            description='The architecture as well as the programming must '
                        'ensure a simple and efficient maintainability.',
            requirement_type=Requirement.TYPE_FUNCTIONAL,
            number='REQ01',
            title='test requirement'
        )

    def test_findings(self):
        query = Smell.objects.filter(smell_title='Subjective Language', requirement=self.requirement).first()
        print(query)
        self.assertIsNotNone(query)


class SuperlativeTest(TestCase):
    def setUp(self) -> None:
        self.detector = SuperlativeSmell()
        self.content = 'The system must provide the signal in the highest resolution ' \
                       'that is desired by the signal customer'

    def test_findings(self):
        findings = self.detector.find(self.content, {})
        self.assertListEqual([8], sorted(findings.keys()))


class DetectionTest(TestCase):
    def test_detection(self):
        content = 'The system must provide the signal in the highest resolution ' \
                  'that is desired by the signal customer'
        content += '\n' + 'The architecture as well as the programming ' \
                          'must ensure a simple and efficient maintainability.'
        findings = run_detection(content)
        self.assertListEqual([8, 27, 29], sorted(findings.keys()))

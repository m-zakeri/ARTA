from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver

from .models import Requirement


@receiver(pre_save, sender=Requirement)
def pre_save_analyze(instance, **kwargs):
    if not instance.id:
        return
    previous_object = Requirement.objects.get(id=instance.id)
    if previous_object.description != instance.description:
        instance.find_smells()


@receiver(post_save, sender=Requirement)
def post_save_analyze(created, instance, **kwargs):
    if created:
        instance.find_smells()

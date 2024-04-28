# Generated by Django 3.0.7 on 2020-06-14 14:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('requirements', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='requirement',
            name='reviewed_date',
        ),
        migrations.AddField(
            model_name='requirement',
            name='smells_count',
            field=models.IntegerField(default=0, verbose_name='count of detected smells'),
        ),
    ]
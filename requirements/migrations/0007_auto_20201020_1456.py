# Generated by Django 3.0.7 on 2020-10-20 11:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('requirements', '0006_auto_20200817_2224'),
    ]

    operations = [
        migrations.AddField(
            model_name='requirement',
            name='cleanness',
            field=models.FloatField(default=0.0, verbose_name='cleanness'),
        ),
        migrations.AddField(
            model_name='requirement',
            name='testability',
            field=models.FloatField(default=0.0, verbose_name='cleanness'),
        ),
    ]

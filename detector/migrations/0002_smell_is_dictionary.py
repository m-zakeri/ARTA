# Generated by Django 3.0.7 on 2020-07-12 21:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detector', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='smell',
            name='is_dictionary',
            field=models.BooleanField(default=False, verbose_name='is dictionary based?'),
        ),
    ]

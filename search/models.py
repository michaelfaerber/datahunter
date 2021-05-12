"""In the models module classes and structures are defined."""

from django.db import models
from django.contrib import admin


class Link(models.Model):
    url = models.URLField(default="", null=True)

class Paper(models.Model):
    paperid = models.URLField(default="", null=True)
    url = models.URLField(default="", null=True)
    title = models.URLField(default="", null=True)

class Dataset(models.Model):
    """Specifies the attributes of a Dataset object."""
    title = models.CharField(max_length=500, default="", null=True)
    description = models.CharField(max_length=5000, default="", null=True)
    topic = models.CharField(max_length=500, default="", null=True)
    link = models.ManyToManyField(Link, default="")
    identifier = models.CharField(max_length=500, default="", null=True)
    creator = models.CharField(max_length=500, default="", null=True)
    publisher = models.CharField(max_length=500, default="", null=True)
    contributor = models.CharField(max_length=500, default="", null=True)
    ACCESS_RIGHT_CHOICES = (('Closed', 'Closed'), ('Open', 'Open'))
    access_right = models.CharField(max_length=50, choices=ACCESS_RIGHT_CHOICES, default="Other",
                                    null=True)
    size = models.DecimalField(decimal_places=2, max_digits=12, null=True)
    DATA_FORMAT_CHOICES = (('zip', 'zip'), ('csv', 'csv'), ('json', 'json'), ('txt', 'txt'),
                           ('xlsx/xls', 'xlsx/xls'))
    data_format = models.CharField(max_length=50, choices=DATA_FORMAT_CHOICES, default="Other",
                                   null=True)
    SOURCE_CHOICES = (('OpenAire', 'OpenAire'), ('Wikidata', 'Wikidata'))
    source = models.CharField(max_length=50, choices=SOURCE_CHOICES, default="Other", null=True)
    issued_date = models.DateField(null=True)
    modified_date = models.DateField(null=True)
    language = models.CharField(max_length=50, default="", null=True)
    ranking_score = models.DecimalField(decimal_places=5, max_digits=10, default=1, null=True)
    referenced_papers = models.CharField(max_length=10000000, default="", null=True)
    referenced_papers = models.ManyToManyField(Paper, default="")
    referenced_papers_string = models.CharField(max_length=10000000, default="", null=True)

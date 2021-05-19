"""The filters module defines django filters, that are used to filter the list of Dataset objects on
   result.html.
   """

import django_filters as dfi
from django import forms
from .models import Dataset


class DatasetFilter(dfi.FilterSet):
    """Defines filter for list of Dataset objects."""
    topic = dfi.CharFilter(lookup_expr='icontains', label="Topic/Keyword:")
    creator = dfi.CharFilter(lookup_expr='icontains', label="Creator/Author:")
    publisher = dfi.CharFilter(lookup_expr='icontains', label="Publisher:")
    ACCESS_RIGHT_CHOICES = (('Closed', 'Closed'), ('Open', 'Open'), ('', 'Any'))
    access_right = dfi.MultipleChoiceFilter(choices=ACCESS_RIGHT_CHOICES,
                                            widget=forms.CheckboxSelectMultiple,
                                            lookup_expr='icontains', label="Access Rights:")
    size = dfi.NumberFilter(lookup_expr='lte', label="Max. Size in kilobyte:")
    DATA_FORMAT_CHOICES = (('zip', 'zip'), ('csv', 'csv'), ('json', 'json'), ('txt', 'txt'),
                           ('xls', 'xlsx/xls'), ('', 'Any'))
    data_format = dfi.MultipleChoiceFilter(choices=DATA_FORMAT_CHOICES,
                                           widget=forms.CheckboxSelectMultiple,
                                           lookup_expr='icontains', label='Data Format:')
    SOURCE_CHOICES = (('OpenAire', 'OpenAire'), ('Wikidata.DB', 'Wikidata.DB'), ('', 'Any'))
    source = dfi.MultipleChoiceFilter(choices=SOURCE_CHOICES,
                                      widget=forms.CheckboxSelectMultiple,
                                      lookup_expr='icontains', label="Source:")
    issued_date = dfi.DateFromToRangeFilter(lookup_expr='icontains',
                                            widget=dfi.widgets.RangeWidget
                                            (attrs={'placeholder': 'yyyy-mm-dd'}),
                                            label='Date of issue:')
    modified_date = dfi.DateFromToRangeFilter(lookup_expr='icontains',
                                              widget=dfi.widgets.RangeWidget
                                              (attrs={'placeholder': 'yyyy-mm-dd'}),
                                              label='Date of last modification:')
    class Meta:
        model = Dataset
        ordering = ['title']
        fields = ['topic', 'creator', 'publisher', 'access_right', 'size', 'data_format',
                  'source', 'issued_date', 'modified_date']

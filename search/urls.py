"""This module defines the urls i.e. links for the search app."""

from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('recommendation', views.recommendation, name='recommendation'),
    path('home', views.home, name='home'),
    path('no_link', views.no_link, name='no_link'),
    path('possible_queries', views.possible_queries, name='possible_queries'),
]

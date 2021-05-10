"""This module defines the urls i.e. links for the search app."""

from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('recommendation', views.recommendation, name='recommendation'),
    path('about', views.about, name='about'),
    path('home', views.home, name='home'),
    path('possible_queries', views.possible_queries, name='possible_queries'),
]

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classify/', views.classify_bird_view, name='classify_bird'),
    path('metrics/', views.train_model_view, name='train_model'),
    path('species/', views.species_directory_view, name='species_directory'),
    path('api/metrics/', views.api_metrics, name='api_metrics'),
]

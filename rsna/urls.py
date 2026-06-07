from django.urls import path
from . import views

app_name = 'rsna'
urlpatterns = [
    path('upload/', views.upload_dicom, name='upload'),
    path('results/<int:prediction_id>/', views.results, name='results'),
]

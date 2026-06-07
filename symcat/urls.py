from django.urls import path
from . import views
app_name = 'symcat'
urlpatterns = [
    path('api/symptoms/', views.get_symptoms, name='get_symptoms'),
    path('api/predict/', views.predict_disease, name='predict_disease'),
    path('symptom-checker/', views.symptom_checker_page, name='symptom_checker'),

]
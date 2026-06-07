from django.urls import path
from . import views

urlpatterns = [
    path('register/',                         views.register_patient, name='register'),
    path('medical-history/<int:patient_id>/', views.medical_history,  name='medical_history'),
    path('success/<int:patient_id>/',         views.success,          name='success'),
    # ✅ NEW — AJAX search for doctor dashboard
    path('search/',                           views.search_json,      name='patients_search_json'),
]
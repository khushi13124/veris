from django.urls import path
from . import views

urlpatterns = [
    # Admin registers a doctor
    path('register/', views.register_doctor, name='register_doctor'),

    # Shows generated credentials after registration
    path('credentials/<str:doctor_id>/', views.doctor_credentials, name='doctor_credentials'),

    # Doctor login / logout
    path('login/', views.doctor_login, name='doctor_login'),
    path('logout/', views.doctor_logout, name='doctor_logout'),

    # First time password change
    path('change-password/', views.change_password, name='change_password'),

    # OTP verification
    path('verify-otp/', views.verify_otp_view, name='verify_otp'),

    # Dashboard
    path('dashboard/', views.dashboard, name='doctor_dashboard'),

    # Edit profile
    path('profile/edit/', views.edit_profile, name='edit_doctor_profile'),

    path('today-queue/', views.today_queue_json, name='today_queue_json'),
    path('patients/list-json/', views.patients_list_json, name='patients_list_json'),
    path('patients/search-json/', views.patients_search_json, name='patients_search_json'),
]


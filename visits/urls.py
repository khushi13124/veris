from django.urls import path
from . import views

app_name = 'visits'

urlpatterns = [
    # Admin / reception
    path('assign/', views.assign_visit, name='assign_visit'),
    path('queue/', views.today_queue, name='today_queue'),

    # Dashboard JSON panel
    path('queue/json/', views.today_queue_json, name='today_queue_json'),

    # Doctor — patient detail
    path('patient/<int:patient_id>/', views.patient_detail, name='patient_detail'),

    # Doctor — visit actions
    path('add/<int:patient_id>/', views.add_visit, name='add_visit'),
    path('update/<int:visit_id>/', views.update_visit, name='update_visit'),
    path('done/<int:visit_id>/',  views.mark_done, name='mark_done'),
    path('status/<int:visit_id>/<str:status>/', views.update_visit_status,  name='update_visit_status'),

    # AI — save symcat result to visit
    path('save-ai/<int:visit_id>/',views.save_ai_result, name='save_ai_result'),
]
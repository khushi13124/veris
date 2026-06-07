from django.apps import AppConfig


class SymcatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'symcat'

    def ready(self):
        from .views import load_model_and_data
        load_model_and_data()
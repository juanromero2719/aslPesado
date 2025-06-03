# example/urls.py
from django.urls import path
from .views import upload_view, healthcheck

urlpatterns = [
    path('', healthcheck, name='healthcheck'),  # ← esta responde rápido
    path('asl/', upload_view, name='asl_upload'),
]
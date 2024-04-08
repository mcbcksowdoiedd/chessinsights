# filehandler/urls.py
from django.urls import path
from .views import upload_file, download_file, download_pdf, upload_success
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('upload/', upload_file, name='upload'),
    path('download/', download_file, name='download_file'),
    path('upload_success/', upload_success, name='upload_success'),
    path('download/pdf/<int:file_id>/', download_pdf, name='download_pdf'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
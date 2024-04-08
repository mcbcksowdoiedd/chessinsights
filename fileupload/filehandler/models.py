from django.db import models
from django.utils import timezone

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    player_color = models.CharField(max_length=10)  # Add player_color field

from django.contrib import admin
from .models import PredictionRecord

@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('predicted_species', 'confidence', 'image_name', 'created_at')
    search_fields = ('predicted_species', 'image_name')

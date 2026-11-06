from django.db import models

class PredictionRecord(models.Model):
    """Stores record of bird identification predictions."""
    image_name = models.CharField(max_length=255)
    predicted_species = models.CharField(max_length=255)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_species} ({self.confidence}%)"

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction Record'
        verbose_name_plural = 'Prediction Records'

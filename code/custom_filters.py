import base64
from django import template

register = template.Library()

@register.filter
def image_to_base64(image):
    image_data = image.read()
    base64_encoded = base64.b64encode(image_data).decode()
    return f"data:image/png;base64,{base64_encoded}"

from django import forms

class BirdImageUploadForm(forms.Form):
    image = forms.ImageField(
        required=True,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'id': 'birdImageInput',
            'accept': 'image/*'
        })
    )

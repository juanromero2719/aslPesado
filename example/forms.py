from django import forms

class ImageUploadForm(forms.Form):
    # el archivo original (opcional si ya mandamos la versión rotada)
    image = forms.ImageField(
        label="Selecciona imagen",
        required=False,
        widget=forms.ClearableFileInput(
            attrs={
                "accept": "image/*",
                "capture": "environment",
            }
        ),
    )
    # aquí llegará la imagen rotada como base-64
    rotated = forms.CharField(widget=forms.HiddenInput(), required=False)

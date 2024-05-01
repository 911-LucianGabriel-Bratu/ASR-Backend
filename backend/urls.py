from django.urls import path

from . import views

urlpatterns = [
    path('upload/', views.upload_wav, name='upload'),
    path('get-image/', views.get_image, name='get-image')
]
import io
import os
import wave

from django.shortcuts import render

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .asr.runner import get_predictions_for_file, get_predictions_for_file_server_running
from .asr.model import brn_model
from django.conf import settings


@csrf_exempt
def upload_wav(request):
    if request.method == 'POST' and request.FILES['file']:
        wav_file = request.FILES['file']
        save_path = os.path.join(settings.BASE_DIR, 'backend', 'asr', 'files')
        file_path = os.path.join(save_path, wav_file.name)
        with open(file_path, 'wb') as destination:
            for chunk in wav_file.chunks():
                destination.write(chunk)
        model_path = os.path.join(settings.BASE_DIR, 'backend', 'asr', 'results', 'model_debug_2.h5')
        prediction = get_predictions_for_file_server_running(file_path, input_to_softmax=brn_model(input_dim=161, units=200),
                    model_path=model_path)
        os.remove(file_path)
        return JsonResponse({'prediction': prediction})
    else:
        return JsonResponse({'error': 'No file uploaded'}, status=400)


@csrf_exempt
def get_image(request):
    if request.method == 'GET':
        image_path = os.path.join(settings.BASE_DIR, 'backend', 'image.jpg')

        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return HttpResponse(image_data, content_type="image/jpeg")
        else:
            return HttpResponse(status=404)

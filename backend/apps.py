from django.apps import AppConfig

from .asr.audio_generator import AudioGenerator


class BackendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend'

    def ready(self):
        from .asr.data_gen import DataGen
        DataGen.instance = AudioGenerator()
        DataGen.instance.load_train_data()
        DataGen.instance.load_validation_data()
        print("Loaded train and test data")

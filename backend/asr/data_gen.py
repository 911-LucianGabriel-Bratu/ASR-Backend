from .utils import int_sequence_to_text
from keras import backend as K
import numpy as np

class DataGen:
    instance = None

def get_predictions_for_file(audio_file, input_to_softmax, model_path):
    # Check if data_gen instance is initialized
    if not DataGen.instance:
        raise Exception("DataGen instance is not initialized. Make sure your server is running correctly.")

    K._get_available_gpus()
    data_point = DataGen.instance.normalize(DataGen.instance.featurize(audio_file))
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()
    return ''.join(int_sequence_to_text(pred_ints))
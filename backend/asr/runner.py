from keras.utils.vis_utils import plot_model

from .data_gen import DataGen
from .audio_generator import AudioGenerator, plot_raw_audio, plot_spectrogram_feature, vis_train_features
from jiwer import wer
import numpy as np
from keras import backend as K

from .model import brn_model
from .utils import int_sequence_to_text
from IPython.display import Audio


def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
    total_words = len(ref_words)
    wer = (substitutions + deletions + insertions) / total_words
    print("Substitutions: " + str(substitutions) + "\n")
    print("Deletions: " + str(deletions) + "\n")
    print("Insertions: " + str(insertions) + "\n")
    print("Total words: " + str(total_words) + "\n")

    return wer * 100


def get_predictions_for_file_server_running(audio_file, input_to_softmax, model_path):
    if not DataGen.instance:
        raise Exception("DataGen instance is not initialized.")

    K._get_available_gpus()
    data_point = DataGen.instance.normalize(DataGen.instance.featurize(audio_file))
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()
    return ''.join(int_sequence_to_text(pred_ints))


def get_predictions_for_file(audio_file, input_to_softmax, model_path):
    K._get_available_gpus()
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    data_point = data_gen.normalize(data_gen.featurize(audio_file))
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()
    return ''.join(int_sequence_to_text(pred_ints))


def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    K._get_available_gpus()
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()

    # obtain the true transcription and the audio features
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()

    # play the audio file, and display the true and predicted transcriptions
    print('-' * 80)
    Audio(audio_path)
    print('True transcription:\n' + '\n' + transcr)
    print('-' * 80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-' * 80)
    print('\nWER: ' + str(wer(transcr, ''.join(int_sequence_to_text(pred_ints))) * 100))


if __name__ == '__main__':
    get_predictions(index=72,
                    partition='validation',
                    input_to_softmax=brn_model(input_dim=161, units=200),
                    model_path='./results/model_debug_2.h5')
    # vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()
    # plot_raw_audio(vis_raw_audio)
    # plot_spectrogram_feature(vis_spectrogram_feature)
    # print(vis_text)
    # input_to_softmax = brn_model(input_dim=161, units=200)
    # input_to_softmax.load_weights('./results/model_debug_2.h5')
    # plot_model(input_to_softmax, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

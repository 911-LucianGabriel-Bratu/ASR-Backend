import keras.optimizers

from audio_generator import AudioGenerator
import _pickle as pickle

from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda)
from keras.optimizers import gradient_descent_v2
from keras.callbacks import ModelCheckpoint
from keras import backend as k

import os

from model import brn_model

SGD = gradient_descent_v2.SGD
k._get_available_gpus()

class SpeechRecognitionTrainer:
    def __init__(self):
        pass

    @staticmethod
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    @staticmethod
    def add_ctc_loss(input_to_softmax):
        the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
        input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
        label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
        output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
        # CTC loss is implemented in a lambda layer
        loss_out = Lambda(SpeechRecognitionTrainer.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [input_to_softmax.output, the_labels, output_lengths, label_lengths])
        model = Model(
            inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
            outputs=loss_out)
        return model

    def train_model(self,
                    input_to_softmax,
                    pickle_path,
                    save_model_path,
                    train_json='train.json',
                    valid_json='valid.json',
                    minibatch_size=20,
                    spectrogram=True,
                    mfcc_dim=13,
                    optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                    #optimizer=adam_v2.Adam(lr=0.01),
                    epochs=100,
                    verbose=1,
                    sort_by_duration=False,
                    max_duration=10.0):
        # create a class instance for obtaining batches of data
        audio_gen = AudioGenerator(minibatch_size=minibatch_size,
                                   spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration,
                                   sort_by_duration=sort_by_duration)
        # add the training data to the generator
        audio_gen.load_train_data(train_json)
        audio_gen.load_validation_data(valid_json)
        # calculate steps_per_epoch
        num_train_examples = len(audio_gen.train_audio_paths)
        steps_per_epoch = num_train_examples // minibatch_size
        # calculate validation_steps
        num_valid_samples = len(audio_gen.valid_audio_paths)
        validation_steps = num_valid_samples // minibatch_size

        # add CTC loss to the NN specified in input_to_softmax
        model = SpeechRecognitionTrainer.add_ctc_loss(input_to_softmax)

        # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

        # make results/ directory, if necessary
        if not os.path.exists('results'):
            os.makedirs('results')

        # add checkpointer
        checkpointer = ModelCheckpoint(filepath='results/' + save_model_path, verbose=0)

        # train the model
        hist = model.fit(x=audio_gen.next_train(),
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         validation_data=audio_gen.next_valid(),
                         validation_steps=validation_steps,
                         callbacks=[checkpointer],
                         verbose=verbose)

        # save model loss
        with open('results/' + pickle_path, 'wb') as f:
            pickle.dump(hist.history, f)


if __name__ == '__main__':
    trainer = SpeechRecognitionTrainer()
    model = brn_model(input_dim=161, units=200)
    input_to_softmax = brn_model
    trainer.train_model(
        input_to_softmax=model,
        pickle_path='model_debug_2.pickle',
        save_model_path='model_debug_2.h5',
        spectrogram=True
    )

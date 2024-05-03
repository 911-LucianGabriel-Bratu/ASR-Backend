import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, GRU, Dropout)


def brn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # =============== 1st Layer =============== #
    # Add bidirectional recurrent layer
    bidirectional_rnn = Bidirectional(
        GRU(units, activation='tanh', return_sequences=True, name='bidir_rnn'))(input_data)
    # Add batch normalization
    batch_normalization = BatchNormalization(name="batch_normalization_bidirectional_rnn")(bidirectional_rnn)
    # Add activation function
    activation = Activation('relu')(batch_normalization)
    # Add dropout
    drop = Dropout(rate=0.1)(activation)

    # =============== 2nd Layer =============== #
    # Add bidirectional recurrent layer
    bidirectional_rnn = Bidirectional(
        GRU(units, activation='tanh', return_sequences=True, name='bidir_rnn'))(drop)
    # Add batch normalization
    batch_normalization = BatchNormalization(name="bn_bidir_rnn_2")(bidirectional_rnn)
    # Add activation function
    activation = Activation('relu')(batch_normalization)
    # Add dropout
    drop = Dropout(rate=0.1)(activation)

    # =============== 3rd Layer =============== #
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(drop)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

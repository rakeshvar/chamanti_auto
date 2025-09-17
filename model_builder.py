import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers, saving
from tensorflow.keras.layers import (Input, Layer, Conv2D, DepthwiseConv2D, MaxPooling2D,
                                     BatchNormalization, LayerNormalization, Activation,
                                     Dense, Bidirectional, LSTM, GRU, Dropout)

from deformer.deformer import Deformer

names = { Conv2D: 'Conv', DepthwiseConv2D: 'DConv', MaxPooling2D: 'Pool', BatchNormalization: 'BatchNorm',
          LayerNormalization: 'LayerNorm', Activation: 'Act', Dense: 'Dense', Bidirectional: 'Bidir',
          LSTM: 'LSTM', GRU: 'GRU', Dropout: 'Dropout'}

@saving.register_keras_serializable()
class CRNNReshape(Layer):
    """ Custom layer to reshape (b, h, w, c) to (b, w, c*h) to convert CNN feature maps to RNN sequence input. """
    def __init__(self, **kwargs):
        super(CRNNReshape, self).__init__(**kwargs)

    def call(self, inputs):
        b, h, w, c = inputs.shape
        x = tf.transpose(inputs, perm=[0, 2, 1, 3])  # (b, w, h, C)
        return tf.reshape(x, [b, w, h*c])

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return b, w, h * c

class CTCLayer(Layer):
    """ You can directly add the loss to the model. But having this class makes the model summary look good. """
    def __init__(self, width_down, **kwargs):
        super().__init__(**kwargs)
        self.width_down = width_down

    def call(self, labels, softmaxout, widths, lengths):
        widths = tf.cast(tf.math.ceil(tf.cast(widths, tf.float32) / self.width_down), tf.int32)
        self.add_loss(keras.backend.ctc_batch_cost(labels, softmaxout, widths, lengths))
        return softmaxout   # Return can be any dummy value

    def get_config(self):
        config = super().get_config()
        config.update({'width_down': self.width_down})
        return config

names[CRNNReshape] = 'CRNNReshape'
names[CTCLayer] = 'CTCLayer'

def get_inference_model(layer_specs, height, num_classes):
    images = Input(shape=(height, None, 1), name="image")

    x = images
    for i, (layer_cls, kwargs) in enumerate(layer_specs):
        name = f"{names.get(layer_cls, layer_cls.__name__)}{i}"
        if layer_cls == Bidirectional:                     # Bidirectional wraps another layer
            rnn_layer = kwargs.pop("layer")
            x = layer_cls(rnn_layer, **kwargs, name=name)(x)
        else:
            x = layer_cls(**kwargs, name=name)(x)


    probabilities = Dense(units=num_classes+1, activation='softmax', name='softmax')(x)

    model = Model(images, probabilities, name="CRNN")
    return model


def get_total_width_pooling(model):
    tot_wd_pooling = 1
    for layer in model.layers:
        if isinstance(layer, MaxPooling2D):
            tot_wd_pooling *= layer.pool_size[1]
    return tot_wd_pooling


def get_train_model(model, deformer_args, learning_rate):
    images = Input(shape=model.input_shape[1:], name="image")
    labeling = Input(name="labeling", shape=(None,), dtype="int32")
    image_width = Input(name="image_width", shape=(1,), dtype="int32")
    labeling_length = Input(name="labeling_length", shape=(1,), dtype="int32")

    deformed_images = Deformer(**deformer_args)(images)
    probabilities = model(deformed_images)
    tot_wd_pooling = get_total_width_pooling(model)
    output = CTCLayer(tot_wd_pooling, name="CTC")(labeling, probabilities, image_width, labeling_length)

    inputs = [images, labeling, image_width, labeling_length]
    ctc_model = Model(inputs=inputs, outputs=output, name="CRNN_CTC")
    ctc_model.compile(optimizer=optimizers.Adam(learning_rate, clipnorm=5.))
    return ctc_model
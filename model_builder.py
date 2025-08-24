import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, Sequential, optimizers, saving
from tensorflow.keras.layers import (Layer, Conv2D, DepthwiseConv2D, MaxPooling2D,
                                     BatchNormalization, LayerNormalization, Activation,
                                     Dense, Bidirectional, LSTM, GRU, Dropout)

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

@saving.register_keras_serializable()
class CTCLayer(Layer):
    """ You can directly add the loss to the model. But having this class makes the model summary look good. """
    def __init__(self, width_down, **kwargs):
        super().__init__(**kwargs)
        self.width_down = width_down

    def call(self, labels, logits, widths, lengths):
        widths //= self.width_down
        self.add_loss(keras.backend.ctc_batch_cost(labels, logits, widths, lengths))
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({'width_down': self.width_down})
        return config

names[CRNNReshape] = 'CRNNReshape'
names[CTCLayer] = 'CTCLayer'

class CRNNBuilder:
    def __init__(self, layer_specs, input_shape, max_label_length, num_classes, learning_rate=3e-4):
        """
        Args:
            layer_specs: list of (layer, kwargs) tuples
            input_shape: (H, W, C) without batch dimension
            num_classes: number of output classes for CTC (incl. blank)
            learning_rate: optimizer LR
        """
        self.layer_specs = layer_specs
        self.input_shape = input_shape
        self.max_label_length = max_label_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        ########################################
        # First build the forward model 
        ########################################
        inputs = layers.Input(shape=self.input_shape, name="image")

        x = inputs
        for i, (layer_cls, kwargs) in enumerate(self.layer_specs):
            name = f"{names[layer_cls]}{i}"
            if layer_cls == layers.Bidirectional:                # Handle recurrent layers specially
                rnn_layer = kwargs['layer']
                if len(x.shape) == 4:                           # CNN -> Reshape -> RNN
                    x = CRNNReshape(name=f"CrnnReshape{i}")(x)  # (B, h, w, C) -> (B, w, C*h)
                x = layer_cls(rnn_layer, name=name)(x)
            else:
                x = layer_cls(**kwargs, name=name)(x)

        outputs = Dense(units=self.num_classes+1, activation='softmax', name='softmax')(x)
        self.model = Model(inputs, outputs, name="CRNN")

        ########################################
        # Then the training model with loss 
        ########################################
        labeling = layers.Input(name="labeling", shape=(max_label_length,), dtype="int32")
        image_width = layers.Input(name="image_width", shape=(1,), dtype="int32")
        labeling_length = layers.Input(name="labeling_length", shape=(1,), dtype="int32")
        _logits = CTCLayer(4, name="CTC")(labeling, self.model.output, image_width, labeling_length)

        self.ctc_model = Model(inputs=[self.model.input, labeling, image_width, labeling_length],
                          outputs=_logits,
                          name="CRNN_CTC")
        self.ctc_model.compile(optimizer=optimizers.Adam(self.learning_rate,
                          clipnorm=5.))

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
        return self.model



from tensorflow.keras.layers import (Conv2D, DepthwiseConv2D, MaxPooling2D,
                                     BatchNormalization, LayerNormalization, Activation,
                                     Dense, Bidirectional, LSTM, GRU, Dropout)

from model_builder import CRNNReshape

balanced_spec = [
    # Stage 1
    (Conv2D, {'filters': 64, 'kernel_size': (3,3), 'padding': 'same'}),   # B × 96 × W × 64
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (Conv2D, {'filters': 64, 'kernel_size': (3,3), 'padding': 'same'}),   # B × 96 × W × 64
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,2)}),                                 # B × 48 × W/2 × 64

    # Stage 2
    (Conv2D, {'filters': 128, 'kernel_size': (3,3), 'padding': 'same'}),  # B × 48 × W/2 × 128
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (Conv2D, {'filters': 128, 'kernel_size': (3,3), 'padding': 'same'}),  # B × 48 × W/2 × 128
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,2)}),                                 # B × 24 × W/4 × 128

    # Stage 3 (horizontal receptive field)
    (Conv2D, {'filters': 256, 'kernel_size': (3,3), 'padding': 'same'}),  # B × 24 × W/4 × 256
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (Conv2D, {'filters': 256, 'kernel_size': (1,5), 'padding': 'same'}),  # B × 24 × W/4 × 256
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,1)}),                                 # B × 12 × W/4 × 256

    # Stage 4 (dilated convs)
    (Conv2D, {'filters': 384, 'kernel_size': (3,3), 'dilation_rate': (1,2), 'padding': 'same'}),  # B × 12 × W/4 × 384
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (Conv2D, {'filters': 384, 'kernel_size': (3,3), 'padding': 'same'}),  # B × 12 × W/4 × 384
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),

    # Stage 5
    (Conv2D, {'filters': 512, 'kernel_size': (3,3), 'padding': 'same'}),  # B × 12 × W/4 × 512
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (Conv2D, {'filters': 512, 'kernel_size': (1,7), 'padding': 'same'}),  # B × 12 × W/4 × 512
    (BatchNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,1)}),                                 # B × 6 × W/4 × 512

    # Collapse height to 1
    (Conv2D, {'filters': 512, 'kernel_size': (6,1), 'strides': (6,1), 'padding': 'valid'}),  # B × 1 × W/4 × 512
    (Conv2D, {'filters': 256, 'kernel_size': (1,1)}),                                        # B × 1 × W/4 × 256

    # Auto Reshape →  (B, W/4, 1, feat=256)

    # Sequence encoder
    (Bidirectional, {'layer': LSTM(256, return_sequences=True)}),       # B × (W/4) × 512
    (Dropout, {'rate': 0.2}),
    (Bidirectional, {'layer': LSTM(256, return_sequences=True)}),       # B × (W/4) × 512
    (Dropout, {'rate': 0.2}),

    # Output projection (CTC logits)
    # (Dense, {'units': num_classes, 'activation': 'linear'})             # B × (W/4) × num_classes
]

lite_spec = [
    # Stage 1
    (Conv2D, {'filters': 16, 'kernel_size': (3,3), 'padding': 'same'}),
    (LayerNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,2)}),

    # Stage 2
    (DepthwiseConv2D, {'kernel_size': (3,3), 'padding': 'same'}),
    (Conv2D, {'filters': 24, 'kernel_size': (1,1)}),  # pointwise
    (LayerNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,2)}),

    # Stage 3
    (DepthwiseConv2D, {'kernel_size': (1,5), 'padding': 'same'}),
    (Conv2D, {'filters': 40, 'kernel_size': (1,1)}),
    (LayerNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,1)}),

    # Stage 4
    (DepthwiseConv2D, {'kernel_size': (3,3), 'padding': 'same'}),
    (Conv2D, {'filters': 64, 'kernel_size': (1,1)}),
    (LayerNormalization, {}),
    (Activation, {'activation': 'swish'}),

    # Stage 5
    (DepthwiseConv2D, {'kernel_size': (1,5), 'padding': 'same'}),
    (Conv2D, {'filters': 96, 'kernel_size': (1,1)}),
    (LayerNormalization, {}),
    (Activation, {'activation': 'swish'}),
    (MaxPooling2D, {'pool_size': (2,1)}),

    # Projection
    (Conv2D, {'filters': 192, 'kernel_size': (1,1)}),

    # Sequence encoder
    (Bidirectional, {'layer': GRU(256, return_sequences=True)}),
    (Dropout, {'rate': 0.2}),

    # CTC head
    # (Dense, {'units': num_classes, 'activation': 'linear'})
]

specs = {'balanced': balanced_spec, 'lite': lite_spec}

import tensorflow as tf
from tensorflow.keras import models, layers
from deformer import Deformer  # import from wherever you placed it

H, W, C = 96, 256, 1
inp = layers.Input(shape=(H, W, C), name="image")
x = Deformer(
    zoom_range=(0.9, 1.1),
    rotation_range=(-0.05, 0.05),
    translation_frac=(0.1, 0.1),
    shear_range=(-0.15, 0.15),
    contrast_range=(0.85, 1.15),
    brightness_delta=(-0.15, 0.15),
    num_cutouts=20,
    max_cutout_hw=(16, 16),
    erase_fraction=0.9
)(inp)
# trivial head
out = layers.Identity(name="id")(x)
mdl = models.Model(inp, out, name="AugTest")

# Dummy batch in [0,1]
import numpy as np
x0 = np.random.rand(4, H, W, C).astype("float32")

y_train = mdl(x0, training=True)
y_eval  = mdl(x0, training=False)

print("Same input, training=True vs False -> should differ:",
      np.mean(np.abs(y_train.numpy() - y_eval.numpy())))

# Save & reload (checks serialization)
mdl.save("/tmp/augtest.keras")
re = tf.keras.models.load_model("/tmp/augtest.keras")
y_train2 = re(x0, training=True)
print("Reloaded model works; mean diff (new randomness expected):",
      np.mean(np.abs(y_train2.numpy() - y_train.numpy())))

# If your images are 0–1 grayscale, the blot value of 1.0 is “white”. If you need black blots, set erase_fraction near 0 and then invert your images (or add a blot_value param and set it).


lite_spec = [
    (Deformer, {   # <-- new top entry
        'zoom_range': (0.85, 1.15),
        'rotation_range': (-0.05, 0.05),
        'translation_frac': (0.1, 0.1),
        'shear_range': (-0.2, 0.2),
        'contrast_range': (0.8, 1.2),
        'brightness_delta': (-0.2, 0.2),
        'num_cutouts': 20,
        'max_cutout_hw': (16, 16),
        'erase_fraction': 0.9,
        'name': 'augment'
    }),

    ...
]

from tensorflow.keras.layers import Layer
from tensorflow.keras import saving
import keras_cv

@saving.register_keras_serializable()
class Deformer(Layer):
    """
    OCR-optimized data augmentation layer using Keras-CV.
    Combines geometric, photometric, and document-specific augmentations.
    Only applies during training (training=True).
    """

    default_config = {
        'zoom_range': 0.03,
        'rotation_range': 0.005,
        'translation_range': 0.05,
        'shear_range': 0.05,
        'gaussian_blur': 0.003,
        'num_cutouts': 20,
        'cutout_size': (.04, .04),
        'contrast_range': 0.5,
        'brightness_range': 0.5,
    }

    def __init__(self,
                 # Geometric parameters
                 zoom_range,
                 rotation_range,
                 translation_range,
                 shear_range,

                 # Noise and artifacts
                 gaussian_blur,
                 num_cutouts,  # number of cutout rectangles
                 cutout_size,  # max cutout size

                 # Photometric parameters
                 contrast_range,  # contrast variation
                 brightness_range,  # brightness variation

                 **kwargs):
        super().__init__(**kwargs)

        # Store parameters
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.shear_range = shear_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.gaussian_blur = gaussian_blur
        self.num_cutouts = num_cutouts
        self.cutout_size = cutout_size

        # Build augmentation pipeline using Keras-CV
        self.augmentation_layers = []

        # Geometric augmentations
        if zoom_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomZoom(zoom_range, fill_mode="nearest"))

        if rotation_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomRotation(rotation_range, fill_mode="nearest"))

        if translation_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomTranslation(translation_range, translation_range, fill_mode="nearest"))

        if shear_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomShear(shear_range, shear_range, fill_mode="nearest"))

        # Noise and artifacts
        if gaussian_blur > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomGaussianBlur(kernel_size=5, factor=gaussian_blur))

        for i in range(num_cutouts):
            self.augmentation_layers.append(
                keras_cv.layers.RandomCutout(*cutout_size, fill_mode="constant", fill_value=i%7)) # 1/7 is black

        # Photometric augmentations
        if contrast_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomContrast(value_range=(0., 1.), factor=contrast_range))

        if brightness_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomBrightness(factor=brightness_range, value_range=(0., 1.)))

    def call(self, images, training=None):
        if not training:
            return images

        x = images
        for layer in self.augmentation_layers:
            x = layer(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'zoom_range': self.zoom_range,
            'rotation_range': self.rotation_range,
            'translation_range': self.translation_range,
            'shear_range': self.shear_range,
            'contrast_range': self.contrast_range,
            'brightness_range': self.brightness_range,
            'gaussian_blur': self.gaussian_blur,
            'num_cutouts': self.num_cutouts,
            'cutout_size': self.cutout_size,
        })
        return config
import tensorflow as tf
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
    
    def __init__(self, 
                 # Geometric parameters
                 zoom_range=0.,           # ±15% zoom
                 rotation_range=0.,       # ±2.9° rotation (in radians)
                 translation_range=0.,    # ±10% translation
                 shear_range=0.,          # shear for text lines
                 
                 # Photometric parameters  
                 contrast_range=0.,       # contrast variation
                 brightness_range=0.,      # brightness variation
                 saturation_range=0.,     # color cast variations
                 
                 # Noise and artifacts
                 gaussian_noise=0.,       # scanner noise
                 num_cutouts=0,             # number of cutout rectangles
                 cutout_size=(.0, .0),        # max cutout size
                 
                 **kwargs):
        super().__init__(**kwargs)
        
        # Store parameters
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range  
        self.translation_range = translation_range
        self.shear_range = shear_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.gaussian_noise = gaussian_noise
        self.num_cutouts = num_cutouts
        self.cutout_size = cutout_size
        
        # Build augmentation pipeline using Keras-CV
        self.augmentation_layers = []
        
        # Geometric augmentations
        if zoom_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomZoom(
                    height_factor=(-zoom_range, zoom_range),
                    width_factor=(-zoom_range, zoom_range)
                )
            )
        if rotation_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomRotation(factor=rotation_range)
            )
        if translation_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomTranslation(
                    height_factor=translation_range,
                    width_factor=translation_range
                )
            )
        if shear_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomShear(
                    x_factor=shear_range,
                    y_factor=0.0  # Only horizontal shear for text
                )
            )
            
        # Photometric augmentations
        if contrast_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomContrast((0., 1.), factor=contrast_range)
            )
        if brightness_range > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomBrightness(factor=brightness_range, value_range=(0., 1.))
            )
        if saturation_range > 0:  # Not for gray-scale images.
            self.augmentation_layers.append(
                keras_cv.layers.RandomSaturation(factor=saturation_range)
            )
            
        # Noise and artifacts
        if gaussian_noise > 0:
            self.augmentation_layers.append(
                keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=(0.0, gaussian_noise))
            )
        if num_cutouts > 0 and cutout_size[0] > 0:
          for i in range(num_cutouts):
            self.augmentation_layers.append(
                keras_cv.layers.RandomCutout(height_factor=cutout_size[0],
                                             width_factor=cutout_size[1],
                                             fill_mode="constant",
                                             fill_value=i%2,
                                             )
                # keras_cv.layers.CutMix(alpha=0.2, seed=None)  # Alternative to cutout
            )

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
            'saturation_range': self.saturation_range,
            'gaussian_noise': self.gaussian_noise,
            'num_cutouts': self.num_cutouts,
            'cutout_size': self.cutout_size,
        })
        return config


# Alternative: RandAugment-based approach for automatic augmentation
@saving.register_keras_serializable()
class DeformerRandAugment(Layer):
    """
    Alternative Deformer using RandAugment for automatic augmentation selection.
    Great for experimentation without manual parameter tuning.
    """
    
    def __init__(self, 
                 magnitude=0.3,             # RandAugment magnitude (0-1)
                 num_layers=3,              # Number of augmentation layers to apply
                 magnitude_stddev=0.1,      # Magnitude variation
                 **kwargs):
        super().__init__(**kwargs)
        
        self.magnitude = magnitude
        self.num_layers = num_layers
        self.magnitude_stddev = magnitude_stddev
        
        # Use RandAugment for automatic augmentation
        self.rand_augment = keras_cv.layers.RandAugment(
            value_range=(0.0, 1.0),
            magnitude=magnitude,
            num_layers=num_layers,
            magnitude_stddev=magnitude_stddev,
            # Exclude augmentations that don't work well for text
            augmentations_per_image=num_layers,
        )
        
        # Add OCR-specific augmentations
        self.ocr_augmentations = tf.keras.Sequential([
            keras_cv.layers.RandomShear(x_factor=0.1, y_factor=0.0),  # Text shear
            keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=(0.0, 0.01)),  # Scanner blur
        ])

    def call(self, images, training=None):
        if not training:
            return images
            
        # Apply RandAugment first
        x = self.rand_augment(images, training=training)
        
        # Then apply OCR-specific augmentations
        x = self.ocr_augmentations(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'magnitude': self.magnitude,
            'num_layers': self.num_layers,
            'magnitude_stddev': self.magnitude_stddev,
        })
        return config


# Lightweight deformer for mobile/edge deployment
@saving.register_keras_serializable()
class DeformerLite(Layer):
    """
    Lightweight deformer with minimal augmentations for mobile/edge deployment.
    """
    
    def __init__(self, intensity=0.1, **kwargs):
        super().__init__(**kwargs)
        self.intensity = intensity
        
        # Minimal augmentation pipeline
        self.augmentations = tf.keras.Sequential([
            keras_cv.layers.RandomRotation(factor=intensity * 0.3),
            keras_cv.layers.RandomTranslation(
                height_factor=intensity * 0.5,
                width_factor=intensity * 0.5
            ),
            keras_cv.layers.RandomContrast(factor=intensity),
            keras_cv.layers.RandomBrightness(factor=intensity * 0.8),
        ])

    def call(self, images, training=None):
        if not training:
            return images
        return self.augmentations(images, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({'intensity': self.intensity})
        return config
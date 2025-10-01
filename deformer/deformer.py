import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


GEOMETRIC_FREQ = 4/5
WARP_FREQ = 1/4
EROSION_FREQ = 1/4
BINARIZE_FREQ = 1/4
JPEG_FREQ = 1/8
LIGHTING_FREQ = 1/4

INK_BLOT_FREQ = 0/5  # Better do it in numpy in Lekhaka
WIDTH_PER_BLOT = 16

default_args = {
     # Geometric parameters
    "zoom_range":0.1,
    "translation_range":(0.1, .05),
    "shear_range":0.02,
    "warp_intensity":0.03,

     # Ink blots
    "ink_blot_fraction": .1, # Number of ink-blots for paper-blot

     # Local transformation parameters
    "erosion_intensity":.05,
    "binarize_threshold":0.1,
    "jpeg_quality_range":(20, 80),

     # Lighting parameters
    "brightness_range":.2,
    "contrast_range":.3,
}

class Deformer(Layer):
    """
    Advanced OCR-optimized data augmentation layer.
    Values: 0=paper, 1=ink, intermediate values possible.
    """

    def __init__(self,
                 zoom_range,
                 translation_range,
                 shear_range,
                 warp_intensity,

                 ink_blot_fraction,
                 erosion_intensity,
                 binarize_threshold,
                 jpeg_quality_range,

                 brightness_range,
                 contrast_range,

                 **kwargs):
        super().__init__(**kwargs)

        self.zoom_range = zoom_range
        self.translation_range = translation_range
        self.shear_range = shear_range
        self.warp_intensity = warp_intensity
        self.ink_blot_fraction = ink_blot_fraction
        self.erosion_intensity = erosion_intensity
        self.binarize_threshold = binarize_threshold
        self.jpeg_quality_range = jpeg_quality_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def compute_output_signature(self, input_signature):
        return input_signature

    def call(self, images, training=True):
        if not training:
            return images

        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        def _mask_apply(x_, freq, fn, *args):
            mask = tf.random.uniform([batch_size, 1, 1, 1]) < freq
            return tf.where(mask, fn(x_, *args), x_)

        # Apply transformations in order
        x = images
        x = _mask_apply(x, GEOMETRIC_FREQ, self._apply_geometric_transforms, height, width)
        x = _mask_apply(x, WARP_FREQ, self._apply_page_warp, height, width)
        x = _mask_apply(x, EROSION_FREQ, self._apply_erosion)
        # x = _mask_apply(x, INK_BLOT_FREQ, self._apply_ink_blots, height, width)
        x = _mask_apply(x, BINARIZE_FREQ, self._apply_binarization)
        x = _mask_apply(x, JPEG_FREQ, self._apply_jpeg_compression)
        x = _mask_apply(x, LIGHTING_FREQ, self._apply_lighting, height, width)
        return tf.clip_by_value(x, 0.0, 1.0)

    def _apply_geometric_transforms(self, images, height, width):
        """Apply geometric transformations: zoom (left-anchored), translation, and shear"""
        batch_size = tf.shape(images)[0]
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)

        # Generate random parameters
        zoom = tf.exp(tf.random.uniform([batch_size], -self.zoom_range, self.zoom_range))

        trh, trw = self.translation_range
        ty = tf.random.uniform([batch_size], -trh * height_f, trh * height_f)
        tx = tf.random.uniform([batch_size], -trw * width_f, 0)  # Only negative for width

        shear_x = tf.random.uniform([batch_size], -self.shear_range, self.shear_range)
        shear_y = tf.random.uniform([batch_size], -self.shear_range, self.shear_range)

        # Build transformation matrices (no rotation, left-anchored zoom)
        # Left margin anchor: cx=0, cy=height/2
        cy = height_f / 2.0

        # Transformation matrix elements
        a11 = zoom + shear_x  # Scale + shear in x
        a12 = shear_x  # Shear component
        a21 = shear_y  # Shear in y
        a22 = zoom + shear_y  # Scale + shear in y

        # Left-anchored zoom translation + user translation
        tx_total = cy * (1 - a22) + tx  # No cx term since cx=0
        ty_total = cy * (1 - a22) + ty

        # Stack into transformation matrices [a11, a12, tx, a21, a22, ty, 0, 0]
        transforms = tf.stack([a11, a12, tx_total, a21, a22, ty_total,
                               tf.zeros([batch_size]), tf.zeros([batch_size])], axis=1)

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            transforms=transforms,
            output_shape=[height, width],
            fill_mode="CONSTANT",
            fill_value=0.0,
            interpolation="BILINEAR"
        )

    def _apply_page_warp(self, images, height, width):
        """Apply page curl/spine curvature using coordinate grid warping"""
        batch_size = tf.shape(images)[0]
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)

        # Create coordinate grids
        y_coords, x_coords = tf.meshgrid(
            tf.linspace(0.0, height_f - 1, height),
            tf.linspace(0.0, width_f - 1, width),
            indexing='ij'
        )

        # Generate warp parameters per batch [batch_size]
        warp_strength = tf.random.uniform([batch_size, 1, 1]) * self.warp_intensity
        warp_frequency = tf.random.uniform([batch_size, 1, 1], 0.5, 2.0)

        # Expand coordinates to batch dimension [batch_size, height, width]
        y_coords_batch = tf.broadcast_to(y_coords[None, :, :], [batch_size, height, width])
        x_coords_batch = tf.broadcast_to(x_coords[None, :, :], [batch_size, height, width])

        # Apply sinusoidal warp (vectorized across batch)
        displacement_x = warp_strength * width_f * tf.sin(warp_frequency * np.pi * y_coords_batch / height_f)
        displacement_y = warp_strength * height_f * tf.sin(warp_frequency * np.pi * x_coords_batch / width_f) * .3

        # Calculate new coordinates
        new_x = tf.clip_by_value(x_coords_batch + displacement_x, 0.0, width_f - 1.001)
        new_y = tf.clip_by_value(y_coords_batch + displacement_y, 0.0, height_f - 1.001)
        warped_coords = tf.stack([new_y, new_x], axis=-1)

        return self._bilinear_sample(images, warped_coords)

    def _bilinear_sample(self, images, coords):
        """Bilinear sampling using gather_nd"""
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        # Extract coordinates
        y_coords = coords[..., 0]
        x_coords = coords[..., 1]

        # Get integer coordinates
        y0 = tf.cast(tf.floor(y_coords), tf.int32)
        x0 = tf.cast(tf.floor(x_coords), tf.int32)
        y1 = tf.minimum(y0 + 1, height - 1)
        x1 = tf.minimum(x0 + 1, width - 1)

        # Get fractional parts and add channel dimension
        fy = tf.expand_dims(y_coords - tf.cast(y0, tf.float32), -1)
        fx = tf.expand_dims(x_coords - tf.cast(x0, tf.float32), -1)

        # Create batch indices
        batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
        batch_idx = tf.tile(batch_idx, [1, height, width])

        # Gather corner values
        def gather_corner(y, x):
            indices = tf.stack([batch_idx, y, x], axis=-1)
            return tf.gather_nd(images, indices)

        I00 = gather_corner(y0, x0)
        I01 = gather_corner(y0, x1)
        I10 = gather_corner(y1, x0)
        I11 = gather_corner(y1, x1)

        # Bilinear interpolation
        I0 = I00 * (1 - fx) + I01 * fx
        I1 = I10 * (1 - fx) + I11 * fx
        return I0 * (1 - fy) + I1 * fy

    def _apply_erosion(self, images):
        """Apply character erosion using random erosion"""
        kernel_size = 1
        kernel = tf.ones([kernel_size, kernel_size, 1])
        eroded = tf.nn.erosion2d(
            images, kernel,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
            dilations = [1, 1, 1, 1]
        )

        # Blend based on erosion strength
        batch_size = tf.shape(images)[0]
        erosion_strength = tf.random.uniform([batch_size, 1, 1, 1]) * self.erosion_intensity

        return images * (1 - erosion_strength) + eroded * erosion_strength

    def _apply_ink_blots(self, images, height, width):
        """Apply random rectangular ink blots - optimized pure TensorFlow version"""
        batch_size = tf.shape(images)[0]
        num_blots = tf.maximum(width // WIDTH_PER_BLOT, 1)
        min_size, max_size = 3, max(height // 15, 5)

        # Generate all blot parameters at once [batch_size, num_blots]
        y1 = tf.random.uniform([batch_size, num_blots], 0, height - max_size, dtype=tf.int32)
        x1 = tf.random.uniform([batch_size, num_blots], 0, width - max_size, dtype=tf.int32)
        blot_h = tf.random.uniform([batch_size, num_blots], min_size, max_size, dtype=tf.int32)
        blot_w = tf.random.uniform([batch_size, num_blots], min_size, max_size, dtype=tf.int32)
        blot_colors = tf.random.uniform([batch_size, num_blots]) < self.ink_blot_fraction
        blot_colors = tf.cast(blot_colors, tf.float32)

        # Create coordinate grids
        y_coords, x_coords = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')

        results = []
        for b in range(batch_size):
            current_image = images[b]

            for i in range(num_blots):
                blot_y1 = y1[b, i]
                blot_x1 = x1[b, i]
                blot_y2 = blot_y1 + blot_h[b, i]
                blot_x2 = blot_x1 + blot_w[b, i]
                color = blot_colors[b, i]

                # Create rectangle mask
                mask = ((y_coords >= blot_y1) & (y_coords < blot_y2) &
                        (x_coords >= blot_x1) & (x_coords < blot_x2))
                mask = tf.cast(mask, tf.float32)[:, :, None]  # Add channel dimension

                # Apply blot: image = image * (1 - mask) + color * mask
                current_image = current_image * (1 - mask) + color * mask

            results.append(current_image)

        return tf.stack(results)

    def _apply_binarization(self, images):
        """Apply random threshold binarization"""
        batch_size = tf.shape(images)[0]

        # Random thresholds per image
        thresholds = tf.random.uniform([batch_size], self.binarize_threshold, 1-self.binarize_threshold)
        thresholds = tf.reshape(thresholds, [batch_size, 1, 1, 1])

        return tf.where(images > thresholds, 1.0, 0.0)

    def _apply_jpeg_compression(self, images):
        """Simulate JPEG compression artifacts"""
        # Simplified JPEG simulation using block averaging + noise
        block_size = 8
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        # Random quality levels
        qualities = tf.random.uniform([batch_size],
                                      self.jpeg_quality_range[0],
                                      self.jpeg_quality_range[1])

        # Block averaging (simplified DCT)
        # Pad to make divisible by block_size
        pad_h = (block_size - height % block_size) % block_size
        pad_w = (block_size - width % block_size) % block_size

        padded = tf.pad(images, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]], 'REFLECT')

        # Reshape to blocks - FIXED: use tf.nn.space_to_depth
        blocked = tf.nn.space_to_depth(padded, block_size)

        # Apply quantization noise based on quality
        noise_scale = (100 - tf.reshape(qualities, [batch_size, 1, 1, 1])) / 100.0 * 0.1
        noise = tf.random.normal(tf.shape(blocked)) * noise_scale
        quantized = blocked + noise

        # Reshape back - FIXED: use tf.nn.depth_to_space
        deblocked = tf.nn.depth_to_space(quantized, block_size)

        # Remove padding
        return deblocked[:, :height, :width, :]

    def _apply_lighting(self, images, height, width):
        """Apply spatially-varying brightness and contrast"""
        batch_size = tf.shape(images)[0]
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)

        # Generate smooth lighting gradients
        y_coords, x_coords = tf.meshgrid(
            tf.linspace(-1.0, 1.0, height),
            tf.linspace(-1.0, 1.0, width),
            indexing='ij'
        )

        brightness = tf.random.uniform([batch_size, 1, 1], -self.brightness_range, self.brightness_range)
        contrast = tf.random.uniform([batch_size, 1, 1], 1.-self.contrast_range, 1.+self.contrast_range)
        angles = tf.random.uniform([batch_size, 1, 1], 0, 2 * np.pi)

        x_coords = tf.expand_dims(x_coords, 0)
        y_coords = tf.expand_dims(y_coords, 0)

        gradient = tf.cos(angles) * x_coords + tf.sin(angles) * y_coords
        brightness_fields = brightness * gradient
        contrast_fields = tf.ones_like(gradient) * contrast

        brightness_fields = tf.expand_dims(brightness_fields, -1) # Channel Dimension
        contrast_fields = tf.expand_dims(contrast_fields, -1)

        return contrast_fields * images + brightness_fields

    def get_config(self):
        config = super().get_config()
        config.update({
            'zoom_range': self.zoom_range,
            'translation_range': self.translation_range,
            'shear_range': self.shear_range,
            'warp_intensity': self.warp_intensity,
            'erosion_intensity': self.erosion_intensity,
            'binarize_threshold': self.binarize_threshold,
            'jpeg_quality_range': self.jpeg_quality_range,
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
        })
        return config

def create_test_model(input_shape, deformer_params):
    inputs = tf.keras.Input(shape=input_shape, name="image")
    deformed = Deformer(**deformer_params, name="deformer")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=deformed, name="DeformerTest")
    return model

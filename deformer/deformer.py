import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import saving
import numpy as np

WIDTH_PER_BLOT = 16

GEOMETRIC_FREQ = 4/5
WARP_FREQ = 1/4
FRAGMENT_FREQ = 1/4
BINARIZE_FREQ = 1/4
JPEG_FREQ = 1/8
LIGHTING_FREQ = 1/4
INK_BLOT_FREQ = 4/5

args = {
     # Geometric parameters
    "zoom_range":0.3,
    "rotation_range":0.03,
    "translation_range":(0.3, .05),
    "shear_range":0.05,
    "warp_intensity":0.05,

     # Ink blots
    "ink_blot_fraction": .5, # Number of ink-blots for paper-blot

     # Local transformation parameters
    "fragment_intensity":0.02,
    "binarize_threshold":0.1,
    "jpeg_quality_range":(20, 80),

     # Lighting parameters
    "brightness_range":0.2,
    "contrast_range":0.3,
}

@saving.register_keras_serializable()
class Deformer(Layer):
    """
    Advanced OCR-optimized data augmentation layer.
    Handles geometric, local, and lighting transformations for single-channel images.
    Values: 0=paper, 1=ink, intermediate values possible.
    """
    
    def __init__(self,
                 zoom_range,
                 rotation_range,
                 translation_range,
                 shear_range,
                 warp_intensity,

                 ink_blot_fraction,
                 fragment_intensity,
                 binarize_threshold,
                 jpeg_quality_range,

                 brightness_range,
                 contrast_range,

                 **kwargs):
        super().__init__(**kwargs)

        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.shear_range = shear_range
        self.warp_intensity = warp_intensity
        self.ink_blot_fraction = ink_blot_fraction
        self.fragment_intensity = fragment_intensity
        self.binarize_threshold = binarize_threshold
        self.jpeg_quality_range = jpeg_quality_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    # @tf.function(jit_compile=True)
    def call(self, images, training=True):
        if not training:
            return images
            
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        # Apply transformations in order
        x = images

        # 1. Geometric transformations (3/4 of time)
        geometric_mask = tf.random.uniform([batch_size]) < GEOMETRIC_FREQ
        x = tf.where(
            tf.reshape(geometric_mask, [batch_size, 1, 1, 1]),
            self._apply_geometric_transforms(x, height, width),
            x
        )
        
        # 2. Page warp (1/2 of time)
        warp_mask = tf.random.uniform([batch_size]) < WARP_FREQ
        x = tf.where(
            tf.reshape(warp_mask, [batch_size, 1, 1, 1]),
            self._apply_page_warp(x, height, width),
            x
        )
        
        # 3. Fragmentation (1/4 of time)
        fragment_mask = tf.random.uniform([batch_size]) < FRAGMENT_FREQ
        x = tf.where(
            tf.reshape(fragment_mask, [batch_size, 1, 1, 1]),
            self._apply_fragmentation(x),
            x
        )
        
        # 4. Ink blots (always applied with random count)
        ink_blot_mask = tf.random.uniform([batch_size]) < INK_BLOT_FREQ
        x = tf.where(
            tf.reshape(ink_blot_mask, [batch_size, 1, 1, 1]),
            self._apply_ink_blots(x, height, width),
            x
        )
        
        # 5. Binarization (1/4 of time)
        binarize_mask = tf.random.uniform([batch_size]) < BINARIZE_FREQ
        x = tf.where(
            tf.reshape(binarize_mask, [batch_size, 1, 1, 1]),
            self._apply_binarization(x),
            x
        )
        
        # 6. JPEG compression (1/8 of time)
        jpeg_mask = tf.random.uniform([batch_size]) < JPEG_FREQ
        x = tf.where(
            tf.reshape(jpeg_mask, [batch_size, 1, 1, 1]),
            self._apply_jpeg_compression(x),
            x
        )
        
        # 7. Lighting (1/4 of time)
        lighting_mask = tf.random.uniform([batch_size]) < LIGHTING_FREQ
        x = tf.where(
            tf.reshape(lighting_mask, [batch_size, 1, 1, 1]),
            self._apply_lighting(x, height, width),
            x
        )
        
        # Clip final output to [0, 1]
        return tf.clip_by_value(x, 0.0, 1.0)
    
    def _apply_geometric_transforms(self, images, height, width):
        """Apply combined geometric transformations using single transformation matrix"""
        batch_size = tf.shape(images)[0]
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        
        # Generate random parameters for each image in batch
        zoom_factors = tf.exp(tf.random.uniform([batch_size], -self.zoom_range, self.zoom_range))
        angles = tf.random.uniform([batch_size], -self.rotation_range, self.rotation_range)

        trh, trw = self.translation_range
        translation_height = tf.random.uniform([batch_size, 1], -trh, trh)
        translation_width = tf.random.uniform([batch_size, 1], -trw, 0)  # Only negative for width
        translations = tf.concat([translation_height, translation_width], axis=1)
        translations = translations * tf.stack([height_f, width_f])
        
        shears = tf.random.uniform([batch_size, 2], -self.shear_range, self.shear_range)
        
        # Build transformation matrices
        transforms = []
        for i in range(batch_size):
            # For Left-margin centered zoom (anchor at x=0, y=height/2)
            cx = width_f / 3.0
            cy = height_f / 2.0
            
            zoom = zoom_factors[i]
            angle = angles[i]
            tx, ty = translations[i][1], translations[i][0]  # Note: TF uses [width, height] order
            shear_x, shear_y = shears[i][0], shears[i][1]
            
            # Combine transformations: T * R * S * Sh * T_inv
            cos_a = tf.cos(angle)
            sin_a = tf.sin(angle)
            
            # Scale + shear matrix
            a11 = zoom * cos_a + shear_x * sin_a
            a12 = -zoom * sin_a + shear_x * cos_a  
            a21 = zoom * sin_a + shear_y * cos_a
            a22 = zoom * cos_a + shear_y * sin_a
            
            # Translation for left-margin centering + user translation
            tx_total = cx * (1 - a11) - cy * a12 + tx
            ty_total = -cx * a21 + cy * (1 - a22) + ty
            
            # Transformation matrix [a11, a12, tx, a21, a22, ty, 0, 0]
            transform = [a11, a12, tx_total, a21, a22, ty_total, 0.0, 0.0]
            transforms.append(transform)
        
        transforms = tf.stack(transforms)
        
        # Apply transformation
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
        
        # Generate warp parameters per batch
        warp_strength = tf.random.uniform([batch_size]) * self.warp_intensity
        warp_frequency = tf.random.uniform([batch_size], 0.5, 2.0)
        
        # Apply sinusoidal warp (spine curvature effect)
        warped_coords = []
        for i in range(batch_size):
            # Horizontal displacement based on vertical position (spine effect)
            displacement_x = warp_strength[i] * width_f * tf.sin(
                warp_frequency[i] * np.pi * y_coords / height_f
            )
            # Slight vertical displacement for page curl
            displacement_y = warp_strength[i] * height_f * 0.3 * tf.sin(
                warp_frequency[i] * np.pi * x_coords / width_f
            )
            
            new_x = tf.clip_by_value(x_coords + displacement_x, 0.0, width_f - 1.001)
            new_y = tf.clip_by_value(y_coords + displacement_y, 0.0, height_f - 1.001)
            
            warped_coords.append(tf.stack([new_y, new_x], axis=-1))
        
        warped_coords = tf.stack(warped_coords)  # [batch, height, width, 2]
        
        # Sample using bilinear interpolation
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
    
    def _apply_fragmentation(self, images):
        """Apply character fragmentation using random erosion"""
        # Small random erosion kernel
        kernel_size = 3
        kernel = tf.ones([kernel_size, kernel_size, 1])
        
        # Random erosion strength per batch
        batch_size = tf.shape(images)[0]
        erosion_strength = tf.random.uniform([batch_size]) * self.fragment_intensity
        
        # Apply conditional erosion
        eroded = tf.nn.erosion2d(
            images, kernel,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
            dilations = [1, 3, 3, 1]
        )
        
        # Blend based on erosion strength
        erosion_strength = tf.reshape(erosion_strength, [batch_size, 1, 1, 1])
        return images * (1 - erosion_strength) + eroded * erosion_strength
    
    def _apply_ink_blots(self, images, height, width):
        """Apply random circular ink blots"""
        batch_size = tf.shape(images)[0]
        width_f = tf.cast(width, tf.float32)
        height_f = tf.cast(height, tf.float32)
        
        # Random number of blots per image
        max_blots = tf.maximum(width // WIDTH_PER_BLOT, 1)
        num_blots = tf.random.uniform([batch_size], 0, max_blots + 1, dtype=tf.int32)
        
        # Generate blob parameters
        max_blots_const = 100  # Reasonable upper bound for vectorization
        
        # Positions and sizes for all possible blots
        blot_positions = tf.random.uniform([batch_size, max_blots_const, 2])
        blot_positions = blot_positions * tf.cast([height, width], tf.float32)
        
        min_radius = 1.0
        max_radius = height_f / 10.0
        blot_radii = tf.random.uniform([batch_size, max_blots_const], min_radius, max_radius)
        
        # Blot colors (0 or 1)
        blot_colors = tf.cast(tf.random.uniform([batch_size, max_blots_const]) < self.ink_blot_fraction, tf.int32)
        blot_colors = tf.cast(blot_colors, tf.float32)
        
        # Create coordinate grids
        y_grid, x_grid = tf.meshgrid(
            tf.cast(tf.range(height), tf.float32),
            tf.cast(tf.range(width), tf.float32),
            indexing='ij'
        )
        y_grid = tf.expand_dims(tf.expand_dims(y_grid, 0), -1)  # [1, H, W, 1]
        x_grid = tf.expand_dims(tf.expand_dims(x_grid, 0), -1)  # [1, H, W, 1]
        
        result = images
        
        # Apply blots (simplified approach - could be optimized further)
        for b in range(batch_size):
            current_image = result[b:b+1]  # [1, H, W, 1]
            
            for i in range(max_blots_const):
                # Check if this blot should be applied
                should_apply = tf.cast(i < num_blots[b], tf.float32)
                
                center_y = blot_positions[b, i, 0]
                center_x = blot_positions[b, i, 1]
                radius = blot_radii[b, i]
                color = blot_colors[b, i]
                
                # Create circular mask
                dist = tf.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
                mask = tf.cast(dist <= radius, tf.float32) * should_apply
                
                # Apply blot
                current_image = current_image * (1 - mask) + color * mask
            
            # Update result
            result = tf.concat([result[:b], current_image, result[b+1:]], axis=0)
        
        return result
    
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
        
        lighting_fields = []
        contrast_fields = []
        
        for i in range(batch_size):
            # Random gradient parameters
            brightness_strength = tf.random.uniform([], -self.brightness_range, self.brightness_range)
            contrast_strength = tf.random.uniform([], 1.0 - self.contrast_range, 1.0 + self.contrast_range)
            
            # Random gradient direction
            angle = tf.random.uniform([], 0, 2 * np.pi)
            
            # Create directional gradient
            gradient = (tf.cos(angle) * x_coords + tf.sin(angle) * y_coords)
            brightness_field = brightness_strength * gradient
            contrast_field = tf.ones_like(gradient) * contrast_strength
            
            lighting_fields.append(brightness_field)
            contrast_fields.append(contrast_field)
        
        brightness_fields = tf.stack(lighting_fields)
        contrast_fields = tf.stack(contrast_fields)
        
        # Expand dimensions to match image shape
        brightness_fields = tf.expand_dims(brightness_fields, -1)
        contrast_fields = tf.expand_dims(contrast_fields, -1)
        
        # Apply lighting: contrast * image + brightness
        return contrast_fields * images + brightness_fields
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'zoom_range': self.zoom_range,
            'rotation_range': self.rotation_range,
            'translation_range': self.translation_range,
            'shear_range': self.shear_range,
            'warp_intensity': self.warp_intensity,
            'fragment_intensity': self.fragment_intensity,
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

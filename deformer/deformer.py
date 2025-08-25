# ----- begin: Deformer layer -----
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Layer
from tensorflow.keras import saving

@saving.register_keras_serializable()
class Deformer(Layer):
    """
    Per-image random geometric + photometric augmentation, plus N rectangular cutouts
    with per-rect independent sizes (bounded by maxima) and erase vs blot fills.

    Runs only when training=True. Serializable into .keras checkpoints.
    """
    def __init__(self,
                 # geometric
                 zoom_range=(0.85, 1.15),         # (min, max) multiplicative
                 rotation_range=(-0.05, 0.05),    # radians
                 translation_frac=(0.10, 0.10),   # (ty_frac, tx_frac) of H and W
                 shear_range=(-0.20, 0.20),       # shear along x in tangent space
                 # photometric
                 contrast_range=(0.8, 1.2),
                 brightness_delta=(-0.2, 0.2),    # add after contrast
                 # cutouts
                 num_cutouts=20,
                 max_cutout_hw=(16, 16),          # (max_h, max_w) in pixels
                 erase_fraction=0.9,              # P(fill=0); P(fill=1)=1-erase_fraction
                 **kwargs):
        super().__init__(**kwargs)
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        self.translation_frac = translation_frac
        self.shear_range = shear_range
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta
        self.num_cutouts = num_cutouts
        self.max_cutout_hw = max_cutout_hw
        self.erase_fraction = erase_fraction

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            zoom_range=self.zoom_range,
            rotation_range=self.rotation_range,
            translation_frac=self.translation_frac,
            shear_range=self.shear_range,
            contrast_range=self.contrast_range,
            brightness_delta=self.brightness_delta,
            num_cutouts=self.num_cutouts,
            max_cutout_hw=self.max_cutout_hw,
            erase_fraction=self.erase_fraction,
        ))
        return cfg

    def build(self, input_shape):
        self.H = int(input_shape[1])
        self.W = int(input_shape[2])

    # -------- helpers --------
    def _rand_scalar(self, low, high, shape=()):
        return tf.random.uniform(shape, low, high, dtype=tf.float32)

    def _batch_affine(self, batch):
        """
        Build a per-image affine transform vector for tfa.image.transform (one per sample).
        We compose: Uncenter -> Affine(zoom, shear, rot, trans) -> Center.
        """
        B = batch
        H = tf.cast(self.H, tf.float32)
        W = tf.cast(self.W, tf.float32)
        cy, cx = (H - 1.) / 2.0, (W - 1.) / 2.0

        # per-image random params
        zoom = self._rand_scalar(self.zoom_range[0], self.zoom_range[1], (B,))
        rot  = self._rand_scalar(self.rotation_range[0], self.rotation_range[1], (B,))
        shy  = self._rand_scalar(self.shear_range[0], self.shear_range[1], (B,))  # shear in x as function of y
        ty   = self._rand_scalar(-self.translation_frac[0]*H, self.translation_frac[0]*H, (B,))
        tx   = self._rand_scalar(-self.translation_frac[1]*W, self.translation_frac[1]*W, (B,))

        c, s = tf.cos(rot), tf.sin(rot)

        # Affine matrix for each sample:
        # [[a0, a1, a2],
        #  [b0, b1, b2],
        #  [ 0,  0,  1]]
        # with shear in x: x' = x + shy*y
        a0 =  zoom * c + 0.0*shy
        a1 = -zoom * (s + shy)
        b0 =  zoom * s
        b1 =  zoom * c

        # center-aware translation:
        # T_total = T(+center) * A * T(-center) + (tx, ty)
        a2 = (-a0*cx - a1*cy) + cx + tx
        b2 = (-b0*cx - b1*cy) + cy + ty

        # tfa.transform expects 8 params (projective last row fixed):
        # [a0, a1, a2, b0, b1, b2, c0, c1], where c0=c1=0 for affine.
        return tf.stack([a0, a1, a2, b0, b1, b2,
                         tf.zeros_like(a0), tf.zeros_like(a0)], axis=1)

    def _per_image_contrast_brightness(self, imgs):
        B = tf.shape(imgs)[0]
        c = self._rand_scalar(self.contrast_range[0], self.contrast_range[1], (B,))
        b = self._rand_scalar(self.brightness_delta[0], self.brightness_delta[1], (B,))
        # vectorized per-image
        def _adj(pair):
            img, i = pair
            img = tf.image.adjust_contrast(img, c[i])
            img = tf.image.adjust_brightness(img, b[i])
            return img
        idx = tf.range(B, dtype=tf.int32)
        return tf.map_fn(_adj, (imgs, idx), dtype=imgs.dtype)

    def _per_image_cutouts(self, imgs):
        """
        Apply N rectangular cutouts per image.
        Each rectangle has independent (h, w) sampled uniformly up to maxima.
        Fill value is 0 (erase) with prob erase_fraction, else 1 (blot).
        """
        B = tf.shape(imgs)[0]
        H = self.H
        W = self.W
        max_h = tf.cast(self.max_cutout_hw[0], tf.int32)
        max_w = tf.cast(self.max_cutout_hw[1], tf.int32)
        K = self.num_cutouts

        Y = tf.range(H, dtype=tf.int32)[:, None]      # (H,1)
        X = tf.range(W, dtype=tf.int32)[None, :]      # (1,W)

        def do_one(img):
            # sample K rectangles
            hs = tf.random.uniform([K], minval=1, maxval=max_h+1, dtype=tf.int32)
            ws = tf.random.uniform([K], minval=1, maxval=max_w+1, dtype=tf.int32)
            y0 = tf.random.uniform([K], minval=0, maxval=H, dtype=tf.int32)
            x0 = tf.random.uniform([K], minval=0, maxval=W, dtype=tf.int32)
            y0 = tf.minimum(y0, H - hs)
            x0 = tf.minimum(x0, W - ws)
            y1 = y0 + hs
            x1 = x0 + ws

            # K×H×W masks
            in_y = (Y[None, :, :] >= y0[:, None, None]) & (Y[None, :, :] < y1[:, None, None])
            in_x = (X[None, :, :] >= x0[:, None, None]) & (X[None, :, :] < x1[:, None, None])
            rects = tf.cast(in_y & in_x, img.dtype)  # (K,H,W)

            # choose erase(0) vs blot(1) per rectangle
            fill_is_one = tf.cast(
                tf.random.uniform([K], 0, 1) > self.erase_fraction, img.dtype
            )  # 1 means blot, 0 means erase

            # two disjoint masks: any erase rect, any blot rect
            erase_mask = tf.reduce_max(rects * (1.0 - fill_is_one)[:, None, None], axis=0)  # (H,W)
            blot_mask  = tf.reduce_max(rects * (fill_is_one)[:, None, None], axis=0)        # (H,W)

            # apply: erase -> set to 0; blot -> set to 1
            c = img.shape[-1]
            erase_mask_c = erase_mask[..., None]
            blot_mask_c  = blot_mask[..., None]
            img = img * (1.0 - erase_mask_c)
            img = img * (1.0 - blot_mask_c) + blot_mask_c * 1.0
            return img

        return tf.map_fn(do_one, imgs, dtype=imgs.dtype)

    # -------- main call --------
    def call(self, images, training=None):
        if not training:
            return images

        images = tf.convert_to_tensor(images)
        images = tf.cast(images, tf.float32)

        B = tf.shape(images)[0]
        # geometric (per-image)
        transforms = self._batch_affine(B)
        images = tfa.image.transform(images, transforms, interpolation="BILINEAR")

        # photometric (per-image)
        images = self._per_image_contrast_brightness(images)

        # cutouts (per-image, many)
        if self.num_cutouts > 0 and self.max_cutout_hw[0] > 0 and self.max_cutout_hw[1] > 0:
            images = self._per_image_cutouts(images)

        # keep range roughly [0,1]
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images
# ----- end: Deformer layer -----

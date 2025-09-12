
configs = {
    'light': {
        'zoom_range': 0.03,
        'rotation_range': 0.005,
        'translation_range': 0.05,
        'shear_range': 0.05,
        'gaussian_blur': 0.003,
        'num_cutouts': 20,
        'cutout_size': (.04, .04),
        'contrast_range': 0.5,
        'brightness_range': 0.5,
    },
    'medium': {
        'zoom_range': 0.06,
        'rotation_range': 0.008,   # ~3 degree
        'translation_range': 0.08,
        'shear_range': 0.10,
        'gaussian_blur': 0.005,
        'num_cutouts': 25,
        'cutout_size': (.06, .06),
        'contrast_range': 0.20,
        'brightness_range': 0.15,
    },
    'heavy': {
        'zoom_range': 0.09,
        'rotation_range': 0.012,
        'translation_range': 0.10,
        'shear_range': 0.15,
        'gaussian_blur': 0.01,
        'num_cutouts': 38,
        'cutout_size': (.08, .08),
        'contrast_range': 0.25,
        'brightness_range': 0.20,
    }
}

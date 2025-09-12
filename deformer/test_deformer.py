#!/usr/bin/env python3
"""
Test script for KerasDeformer augmentation layer.
Generates sample images using Lekhaka and shows before/after augmentation.
"""

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

try:
    import telugu as lang
    from Lekhaka import Scribe, DataGenerator
except ImportError:
    import Lekhaka.telugu as lang
    from Lekhaka.Lekhaka import Scribe, DataGenerator

from deformer import Deformer

def create_test_model(input_shape, deformer_params):
    inputs = tf.keras.Input(shape=input_shape, name="image")
    deformed = Deformer(**deformer_params, name="deformer")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=deformed, name="DeformerTest")
    return model

def plot_comparison(original_batch, deformed_batch, batch_labels, save_path=None):
    batch_size = min(8, original_batch.shape[0])  # Show max 8 samples
    
    fig, axes = plt.subplots(2, batch_size, figsize=(2*batch_size, 4))
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # Original image (top row)
        axes[0, i].imshow(original_batch[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        
        # Deformed image (bottom row)  
        axes[1, i].imshow(deformed_batch[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()

def decode_label(label_array):
    """Convert label array to readable text using language symbols"""
    # Remove padding (assuming 0 is padding)
    label_array = label_array[label_array != 0]
    if len(label_array) == 0:
        return "<empty>"
    
    try:
        # Convert indices to characters
        text = ''.join([lang.symbols[i] if i < len(lang.symbols) else '?' for i in label_array])
        return text[:20]  # Truncate for display
    except:
        return f"<label:{label_array[:5]}...>"

def test_deformer_variants():
    """Test different deformer configurations"""
    variants = {
        'light': {
            'zoom_range': 0.05,
            'rotation_range': 0.02,
            'translation_range': 0.05,
            'shear_range': 0.08,
            'contrast_range': 0.15,
            'brightness_range': 0.10,
            'gaussian_noise': 0.003,
            'num_cutouts': 3,
            'cutout_size': (.04, .04)
        },
        'medium': {
            'zoom_range': 0.10,
            'rotation_range': 0.03,
            'translation_range': 0.08,
            'shear_range': 0.10,
            'contrast_range': 0.20,
            'brightness_range': 0.15,
            'gaussian_noise': 0.005,
            'num_cutouts': 5,
            'cutout_size': (.06, .06)
        },
        'heavy': {
            'zoom_range': 0.15,
            'rotation_range': 0.05,
            'translation_range': 0.10,
            'shear_range': 0.15,
            'contrast_range': 0.25,
            'brightness_range': 0.20,
            'gaussian_noise': 0.01,
            'num_cutouts': 8,
            'cutout_size': (.08, .08)
        }
    }
    
    return variants

def main():
    parser = argparse.ArgumentParser(description="Test KerasDeformer augmentation layer")
    parser.add_argument("-H", "--height", type=int, default=64, help="Image height")
    parser.add_argument("-C", "--chars_per_sample", type=int, default=6, help="Characters per sample")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("-O", "--output_dir", type=str, default="deformer_test_output/",
                       help="Output directory for saved images")
    parser.add_argument("--no_display", action="store_true", help="Don't display images (save only)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Image height: {args.height}, Chars per sample: {args.chars_per_sample}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 60)
    
    scribe_args = {
        'height': args.height,
        'hbuffer': 5,
        'vbuffer': 0,
        'nchars_per_sample': args.chars_per_sample
    }
    
    scriber = Scribe(lang, **scribe_args)
    identity = lambda x: x
    datagen = DataGenerator(scriber, identity, identity, args.batch_size)
    
    print(f"Scriber setup: {scriber}")
    print(f"Max width: {scriber.width}, Input shape: ({args.height}, {scriber.width}, 1)")
    
    # Get deformer parameters
    variants = test_deformer_variants()

    # Run tests
    for deformer_name, deformer_params in variants.items():
        print(f"\nTesting {deformer_name}")
        print(f"Deformer parameters: {deformer_params}")
        print("-" * 60)

        # Create test model
        input_shape = (args.height, scriber.width, 1)
        test_model = create_test_model(input_shape, deformer_params)

        print("Model summary:")
        test_model.summary()
        print("-" * 60)

        # Get original batch from Lekhaka
        images, labels, image_lengths, label_lengths = datagen.get()
        images = 1-images
        print(f"Generated batch shape: {images.shape}")
        print(f"Sample labels: {[decode_label(labels[i][:label_lengths[i]]) for i in range(min(3, len(labels)))]}")

        # Test without augmentation (training=False)
        no_aug_output = test_model(images, training=False)
        print("No augmentation test passed ✓")

        # Test with augmentation (training=True)
        aug_output = test_model(images, training=True)
        print("Augmentation test passed ✓")

        # Verify shapes are preserved
        assert images.shape == aug_output.shape, f"Shape mismatch: {images.shape} vs {aug_output.shape}"
        print(f"Shape preservation verified ✓")

        # Check value ranges
        print(f"Original range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Augmented range: [{tf.reduce_min(aug_output):.3f}, {tf.reduce_max(aug_output):.3f}]")

        # Save comparison plot
        save_path = output_dir / f"test_{deformer_name}.png"

        if not args.no_display:
            print("Displaying comparison...")
            plot_comparison(images, aug_output.numpy(), labels, save_path)
        else:
            # Save without display
            fig, axes = plt.subplots(2, min(8, args.batch_size), figsize=(16, 4))
            if min(8, args.batch_size) == 1:
                axes = axes.reshape(2, 1)

            for i in range(min(8, args.batch_size)):
                axes[0, i].imshow(images[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Original\n{decode_label(labels[i])}', fontsize=8)
                axes[0, i].axis('off')

                axes[1, i].imshow(aug_output[i].squeeze(), cmap='gray')
                axes[1, i].set_title('Augmented', fontsize=8)
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison to: {save_path}")
    
    print(f"\n✅ All tests completed successfully!")
    print(f"Output saved to: {output_dir}")
    
    # Test serialization
    print("\nTesting model serialization...")
    model_path = output_dir / f"test_deformer.keras"
    test_model.save(model_path)
    
    # Load and test
    loaded_model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'Deformer': Deformer}
    )
    
    # Quick test of loaded model
    test_output = loaded_model(images, training=True)
    print(f"✅ Serialization test passed! Model saved to: {model_path}")
    
    print("\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    main()
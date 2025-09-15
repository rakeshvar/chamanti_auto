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

from deformer_basic import Deformer
from deformer_configs import configs as variants

def create_test_model(input_shape, deformer_params):
    inputs = tf.keras.Input(shape=input_shape, name="image")
    deformed = Deformer(**deformer_params, name="deformer")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=deformed, name="DeformerTest")
    return model

def plot_comparison(original_batch, deformed_batch, save_path=None, display=True):
    batch_size = min(8, original_batch.shape[0])  # Show max 8 samples
    fig, axes = plt.subplots(batch_size, 2, figsize=(2, batch_size))

    for i in range(batch_size):
        # Original image (top row)
        axes[i, 0].imshow(original_batch[i].squeeze(), cmap='gray')
        axes[i, 0].axis('off')
        
        # Deformed image (bottom row)  
        axes[i, 1].imshow(deformed_batch[i].squeeze(), cmap='gray')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")

    if display:
        plt.show()


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

        print("Displaying comparison...")
        plot_comparison(images, aug_output.numpy(), save_path, not args.no_display)

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
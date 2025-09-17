import sys
from pathlib import Path
from tensorflow.keras.models import load_model

from model_builder import get_inference_model
from specs import specs


# ------------------------
# Models
# ------------------------
src_ckpt = Path(sys.argv[1])
print(f"Loading model from {src_ckpt}")
src_model = load_model(src_ckpt)
height = src_model.input_shape[1]
num_classes = src_model.output_shape[-1] - 1

spec = specs["lite"]
tgt_model = get_inference_model(spec, height, num_classes)

# ------------------------
# Transfer
# ------------------------
def layers_with_weights(model):
    return  [layer for layer in model.layers if len(layer.get_weights()) > 0]

src_layers = layers_with_weights(src_model)
tgt_layers = layers_with_weights(tgt_model)

transferred = 0
for src_layer, tgt_layer in zip(src_layers, tgt_layers):
    src_weights = src_layer.get_weights()
    tgt_weights = tgt_layer.get_weights()

    # Check if shapes match
    if len(src_weights) == len(tgt_weights):
        shapes_match = all(ow.shape == nw.shape for ow, nw in zip(src_weights, tgt_weights))
        if shapes_match:
            tgt_layer.set_weights(src_weights)
            transferred += 1
            print(f"✓ Transferred {src_layer.name} -> {tgt_layer.name}")
        else:
            print(f"✗ Shape mismatch {src_layer.name} -> {tgt_layer.name}")
        for i, (ow, nw) in enumerate(zip(src_weights, tgt_weights)):
            print(f"  Weight {i}: {ow.shape} vs {nw.shape}")
    else:
        print(f"✗ Weight count mismatch {src_layer.name} -> {tgt_layer.name}")

print(f"Successfully transferred weights for {transferred}/{len(src_layers)} layers")

# ------------------------
# Save
# ------------------------
tgt_ckpt = src_ckpt.with_stem('Trns-' + src_ckpt.stem)
tgt_model.save(tgt_ckpt)
print(f"Saved model to {tgt_ckpt}")

import sys
from pathlib import Path
from tensorflow.keras.models import load_model

from model_builder import get_prediction_model
from specs import specs

src_ckpt = Path(sys.argv[1])

# Old model
print(f"Loading model from {src_ckpt}")
old_model = load_model(src_ckpt)

# New model
spec = specs["lite"]
height = old_model.input_shape[1]
num_classes = old_model.output_shape[-1] - 1
new_model = get_prediction_model(spec, height, num_classes)

# ------------------------
# Transfer
# ------------------------
def layers_with_weights(model):
    return  [layer for layer in model.layers if len(layer.get_weights()) > 0]

old_layers = layers_with_weights(old_model)
new_layers = layers_with_weights(new_model)

transferred = 0
for old_layer, new_layer in zip(old_layers, new_layers):
    old_weights = old_layer.get_weights()
    new_weights = new_layer.get_weights()

    # Check if shapes match
    if len(old_weights) == len(new_weights):
        shapes_match = all(ow.shape == nw.shape for ow, nw in zip(old_weights, new_weights))
        if shapes_match:
            new_layer.set_weights(old_weights)
            transferred += 1
            print(f"✓ Transferred {old_layer.name} -> {new_layer.name}")
        else:
            print(f"✗ Shape mismatch {old_layer.name} -> {new_layer.name}")
        for i, (ow, nw) in enumerate(zip(old_weights, new_weights)):
            print(f"  Weight {i}: {ow.shape} vs {nw.shape}")
    else:
        print(f"✗ Weight count mismatch {old_layer.name} -> {new_layer.name}")

print(f"Successfully transferred weights for {transferred}/{len(old_layers)} layers")

# ------------------------
# Save
# ------------------------
out_ckpt = src_ckpt.with_stem('Trns-' + src_ckpt.stem)
new_model.save(out_ckpt)
print(f"Saved model to {out_ckpt}")

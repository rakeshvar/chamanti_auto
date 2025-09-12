import sys
from pathlib import Path
from tensorflow.keras.models import load_model

from model_builder import CRNNBuilder, CRNNReshape, CTCLayer
from specs import specs

# ------------------------
# Old model
# ------------------------
checkpoint_path = Path(sys.argv[1])
print(f"Loading model from {checkpoint_path}")
old_model = load_model(checkpoint_path, compile=True,
                       custom_objects={'CRNNReshape': CRNNReshape, 'CTCLayer': CTCLayer})

# Get dimensions from loaded model
inputs_shape = old_model.input_shape
image_shape = inputs_shape[0][1:]
max_label_len = inputs_shape[1][1]
num_classes = old_model.get_layer('softmax').units - 1

# ------------------------
# New model
# ------------------------
spec = specs["lite"]
builder = CRNNBuilder(spec, image_shape, max_label_len, num_classes, learning_rate=3e-3)
ctc_model = builder.ctc_model

# ------------------------
# Transfer
# ------------------------
def layers_with_weights(model):
    return  [layer for layer in model.layers if len(layer.get_weights()) > 0]

old_layers = layers_with_weights(old_model)
new_layers = layers_with_weights(ctc_model)

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
ckpt_path = "transfered.keras"
ctc_model.save(ckpt_path)
print(f"Saved model to {ckpt_path}")

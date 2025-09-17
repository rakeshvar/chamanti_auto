import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from deformer.deformer import default_args as deformer_args
from model_builder import  get_train_model, get_inference_model
from post_process import PostProcessor
from specs import specs

try:
  import telugu as lang
  from Lekhaka import Scribe, Deformer, Noiser
  from Lekhaka import DataGenerator
except ImportError:                          # ModuleNotFoundError:
  import Lekhaka.telugu as lang
  from Lekhaka.Lekhaka import Scribe, Deformer, Noiser
  from Lekhaka.Lekhaka import DataGenerator


# ------------------------
# Parse arguments
# ------------------------
parser = argparse.ArgumentParser(description="Train CRNN with CTC loss",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("init_from", type=str, nargs='?', default="lite",
                    help="Which spec to use if init_from=spec (balanced|lite)")
parser.add_argument("-E", "--num_epochs", type=int, default=100)
parser.add_argument("-S", "--steps_per_epoch", type=int, default=100)
parser.add_argument("-B", "--batch_size", type=int, default=64)
parser.add_argument("-O", "--output_dir", type=str, default="models/")
parser.add_argument("-L", "--learning_rate", type=float, default=3e-4)

parser.add_argument("-H", "--height", type=int, default=96)
parser.add_argument("-C", "--chars_per_sample", type=int, default=6)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
batch_size = args.batch_size
init_from_ckpt = args.init_from.endswith('.keras')

# ------------------------
# Build model
# ------------------------
if init_from_ckpt:
    checkpoint_path = Path(args.init_from)
    print(f"Loading model from {checkpoint_path}")
    inference_model = load_model(checkpoint_path)
    ckpt_with = Path(args.init_from).stem.split('-')[0]

else:
    spec = specs[args.init_from]
    inference_model = get_inference_model(spec, args.height, len(lang.symbols))
    ckpt_with = args.init_from
    # Bias towards blanks
    sml = inference_model.get_layer("softmax")
    weights, biases = sml.get_weights()
    biases[-1] += 5.5
    sml.set_weights([weights, biases])

inference_model.summary()
ctc_model = get_train_model(inference_model, deformer_args, args.learning_rate)
ctc_model.summary()

# ------------------------
# Language Data Gen
# ------------------------
print(args)
height = inference_model.input_shape[1]

scribe_args = {'height': height, 'hbuffer': 5, 'vbuffer': 0, 'nchars_per_sample': args.chars_per_sample}
scriber = Scribe(lang, **scribe_args)
max_width = scriber.width
print(scriber)

noiser = Noiser(scriber.width//16, .9, 1, height//12)
datagen = DataGenerator(scriber, noiser=noiser, batch_size=batch_size)
max_label_len = datagen.labelswidth

postprocessor = PostProcessor(lang.symbols)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# ------------------------
# Data pipeline (infinite dataset)
# ------------------------
def proper_x_y(images, labels, image_lengths, label_lengths):
    # images = tf.transpose(images, perm=[0, 2, 1, 3])  # (B, H, W, C)
    return {
        'image': images,
        'labeling': labels,
        'image_width': image_lengths,
        'labeling_length': label_lengths
    }, labels  # dummy target

dataset = tf.data.Dataset.from_generator(
    datagen.generator,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, height, max_width, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, datagen.labelswidth), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)
    )
).prefetch(tf.data.AUTOTUNE).map(proper_x_y)

# ------------------------
# Test & Checkpoint
# ------------------------
from datetime import datetime
timestamp = datetime.now().strftime("%m%d-%H%M")
ckpt_head = f"{ckpt_with}-{timestamp}"

class MyCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        image, labels, image_lengths, label_lengths = datagen.get()
        probabilities = inference_model.predict(image)
        probs_lengths = np.ceil(image_lengths / (max_width // probabilities.shape[-2])).astype(int)
        ederr = postprocessor.show_batch(image, image_lengths, labels, label_lengths, probabilities, probs_lengths)
        print(f"Edit Distance Error Rate: {ederr:.1%}")

        ckpt_path = output_dir / (ckpt_head + f"-ep{epoch:03d}-er{int(1000*ederr):03d}.keras")
        inference_model.save(ckpt_path)
        print(f"Saved model to {ckpt_path}")

# ------------------------
# Train
# ------------------------
print("Starting training...")
ctc_model.fit(
    dataset,
    epochs=args.num_epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=[MyCallback()])
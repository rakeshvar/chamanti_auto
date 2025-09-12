import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from model_builder import CRNNBuilder, CRNNReshape, CTCLayer  # <-- from our earlier class
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
# Default Arguments
# ------------------------
elastic_args0 = { 'translation': 0, 'zoom': .0, 'elastic_magnitude': 0, 'sigma': 1, 'angle': 0, 'nearest': True}
elastic_args = { 'translation': 5, 'zoom': .15, 'elastic_magnitude': 0, 'sigma': 30, 'angle': 3, 'nearest': True}
noise_args = { 'num_blots': 25, 'erase_fraction': .9, 'minsize': 4, 'maxsize': 9}
scribe_args = { 'height': 0, 'hbuffer': 5, 'vbuffer': 0, 'nchars_per_sample': 0}

# ------------------------
# Parse arguments
# ------------------------
parser = argparse.ArgumentParser(description="Train CRNN with CTC loss", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-I", "--init_from", type=str, default="lite", help="Which spec to use if init_from=spec (balanced|lite)")
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
height = args.height
scribe_args["height"] = height
scribe_args['nchars_per_sample'] = args.chars_per_sample
noise_args['num_blots'] = height//3

# ------------------------
# Language Data Gen
# ------------------------
printer = PostProcessor(lang.symbols)
scriber = Scribe(lang, **scribe_args)
deformer = Deformer(**elastic_args)
noiser = Noiser(**noise_args)
datagen = DataGenerator(scriber, deformer, noiser, batch_size)

num_classes = len(lang.symbols)
max_width = scriber.width
height = scribe_args["height"]
max_label_len = datagen.labelswidth
input_shape = (height, None, 1)

print(scriber)
print(f"Input shape B, (Ht, Wd, 1) : {batch_size} {input_shape}")
print(args)
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
# Build model
# ------------------------
if args.init_from.endswith('.keras'):
    checkpoint_path = Path(args.init_from)
    print(f"Loading model from {checkpoint_path}")
    ctc_model = load_model(checkpoint_path, compile=True,
                    custom_objects={'CRNNReshape': CRNNReshape, 'CTCLayer': CTCLayer}) #                        'KerasDeformer': KerasDeformer,                        'RandomShear': RandomShear,                        'RandomPerspective': RandomPerspective
    prediction_model = keras.models.Model(ctc_model.get_layer(name="image").output,
                                          ctc_model.get_layer(name="softmax").output)
    ckpt_with = checkpoint_path.stem[:2] + "-cont-"


elif args.init_from in specs:
    spec = specs[args.init_from]
    builder = CRNNBuilder(spec, input_shape, max_label_len, num_classes, learning_rate=args.learning_rate)
    ctc_model = builder.ctc_model
    prediction_model = builder.model
    ckpt_with = args.init_from

    # Bias towards blanks and against punctuation
    sml = ctc_model.get_layer("softmax")
    weights, biases = sml.get_weights()
    biases[-1] += 5.5  # 5-4.6, 6-3.7 , 7-3.6, 8-4
    biases[1:42] -= .5  # 6: .5-3.5  1.5-3.5
    sml.set_weights([weights, biases])

else:
    raise ValueError("Did not understand --init_from. Needs to be name of a spec or a checkpoint.")

ctc_model.summary()


# ------------------------
# Callbacks + Custom Callback: EDER + sample display + checkpoint
# ------------------------
from datetime import datetime
timestamp = datetime.now().strftime("%m%d-%H%M")
ckpt_head = f"{ckpt_with}-{timestamp}"

class MyCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        image, labels, image_lengths, label_lengths = datagen.get()
        print(image.shape)
        probabilities = prediction_model.predict(image)
        probs_lengths = image_lengths // (max_width // probabilities.shape[-2])
        ederr = printer.show_batch(image, image_lengths, labels, label_lengths, probabilities, probs_lengths)
        print(f"Edit Distance Error Rate: {ederr:.1%}")

        name = ckpt_head + f"-ep{epoch:03d}-er{int(1000*ederr):03d}.keras"
        ckpt_path = output_dir / name
        self.model.save(ckpt_path)
        print(f"Saved model to {ckpt_path}")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="loss",  # since no val_loss
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.TensorBoard(log_dir=str(output_dir / "logs"),
                                profile_batch='2,5'),
    MyCallback()
]

# ------------------------
# Train
# ------------------------
print("Starting training...")

tf.profiler.experimental.start(str(output_dir / "logs"))

ctc_model.fit(
    dataset,
    epochs=args.num_epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=callbacks)

# --- Profiling stop ---
tf.profiler.experimental.stop()

import math

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
import editdistance

from deformer.deformer import Deformer
from model_builder import CRNNReshape, CTCLayer
from post_process import PostProcessor

try:
    import telugu as lang
    from Lekhaka import Scribe, DataGenerator
except ModuleNotFoundError:
    import Lekhaka.telugu as lang
    from Lekhaka.Lekhaka import Scribe, DataGenerator

# -----------------------------
# Helper: build datagen
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-C", "--checkpoint", required=False, help="Path to trained .keras model checkpoint", default="models/lite-0824-0956-ep055-er076.keras")
parser.add_argument("-S", "--chars_per_sample", type=int, default=6)
parser.add_argument("-B", "--batch_size", type=int, default=1)
args = parser.parse_args()
batch_size = args.batch_size

# -----------------------------
# Models
# -----------------------------
checkpoint = Path(args.checkpoint)
print(f"Loading model from {checkpoint}")
ctc_model = keras.models.load_model(
    checkpoint,
    compile=False,
    custom_objects={"CRNNReshape": CRNNReshape,
                    "CTCLayer": CTCLayer,
                    "Deformer": Deformer}
)
prediction_model = keras.models.Model(
    ctc_model.get_layer(name="image").output,
    ctc_model.get_layer(name="softmax").output
)

img_shape = prediction_model.get_layer("image").output.shape
print("Image shape: ", img_shape)
_, height, wd_in, ch_in = img_shape

out_shape = prediction_model.get_layer("softmax").output.shape
_, wd_out, num_classes = out_shape
print("Out shape: ", out_shape)

# -----------------------------
# Datagen, Printer
# -----------------------------
scribe_args = {"height": height, "hbuffer": 5, "vbuffer": 0, "nchars_per_sample": args.chars_per_sample}
dummy = lambda x: x
scriber = Scribe(lang, **scribe_args)
datagen = DataGenerator(scriber, dummy, dummy, batch_size)
printer = PostProcessor(lang.symbols)

def one_batch():
    images, labels, img_lens, lbl_lens = datagen.get()
    probs = prediction_model.predict(images)
    wd_scaled_down_by = images.shape[-2] // probs.shape[-2]
    prob_lens = [math.ceil(img_len / wd_scaled_down_by) for img_len in img_lens]
    beams = printer.beam_decode(probs, prob_lens)

    print(lbl_lens)

    i = 0
    for label, image, img_len, lbl_len, prob, prob_len in zip(labels, images, img_lens, lbl_lens, probs, prob_lens):
        label = label[:lbl_len]
        image = image.squeeze()
        image2 = image[::2, :img_len:2]
        prob2 = prob[:prob_len, :]
        shown_chars = printer.labels_to_chars(label)
        seen_labels = printer.decode(prob2)
        greedy_decoded = printer.labels_to_chars(seen_labels)
        nmissed = len(set(label)-set(seen_labels))
        beam_decoded = beams[i]
        eddist = editdistance.eval(label, seen_labels)

        print(i)
        if eddist > -1:
            print("EDIT DISI: ", eddist)
            printer.show_all(label, image2, prob2, True)
            img255 = as255(1 - image)
            prob255 = dilate(as255(prob.T), wd_scaled_down_by, wd_scaled_down_by)
            img = np.vstack((img255, prob255))
            img = Image.fromarray(img)
            img.show()

        i += 1


def as255(v):
    return (255*(v-v.min())/(v.max()-v.min())).astype('uint8')


def dilate(a, dh, dw):
    h, w = a.shape                     # H ⨉ W
    a1 = np.stack((a,)*dh, axis=1)     # H ⨉ Dh ⨉ W
    a2 = np.stack((a1,)*dw, axis=-1)   # H ⨉ Dh ⨉ W ⨉ Dw
    return a2.reshape((h*dh, w*dw))    # HDh ⨉ WDw


while True:
    one_batch()
    try:
        input("Want to see one more? Press Enter. Else Ctrl-Z.")
    except KeyboardInterrupt:
        break

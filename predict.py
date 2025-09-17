import math

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
import editdistance

from model_builder import CRNNReshape
from post_process import PostProcessor

try:
    import telugu as lang
    from Lekhaka import Scribe, DataGenerator
except ModuleNotFoundError:
    import Lekhaka.telugu as lang
    from Lekhaka.Lekhaka import Scribe, DataGenerator

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to trained .keras model checkpoint")
parser.add_argument("-C", "--chars_per_sample", type=int, default=6)
parser.add_argument("-B", "--batch_size", type=int, default=32)
args = parser.parse_args()
batch_size = args.batch_size

# -----------------------------
# Models
# -----------------------------
checkpoint = Path(args.checkpoint)
print(f"Loading model from {checkpoint}")
inference_model = keras.models.load_model(checkpoint, custom_objects={"CRNNReshape": CRNNReshape})
height = inference_model.input_shape[1]
print("Height: ", height)

# -----------------------------
# Datagen, Printer
# -----------------------------
scribe_args = {"height": height, "hbuffer": 5, "vbuffer": 0, "nchars_per_sample": args.chars_per_sample}
scriber = Scribe(lang, **scribe_args)
datagen = DataGenerator(scriber, batch_size=batch_size)
postprocessor = PostProcessor(lang.symbols)

def one_batch():
    images, labels, img_lens, lbl_lens = datagen.get()
    print("Label Lengths: ", lbl_lens)

    probs = inference_model.predict(images)
    wd_scaled_down_by = images.shape[-2] // probs.shape[-2]
    prob_lens = [math.ceil(img_len / wd_scaled_down_by) for img_len in img_lens]
    beams = postprocessor.beam_decode(probs, prob_lens)

    for image, img_len, label, lbl_len, prob, prob_len, beam in zip(images, img_lens, labels, lbl_lens, probs, prob_lens, beams):
        label = label[:lbl_len]
        prob2 = prob[:prob_len, :]
        seen_labels = postprocessor.decode(prob2)

        # Show Characters
        shown_chars = postprocessor.labels_to_chars(label)
        greedy_decoded = postprocessor.labels_to_chars(seen_labels)
        print(shown_chars)
        print(greedy_decoded)
        print(beam)

        eddist = editdistance.eval(label, seen_labels)
        print("EDIT DISI: ", eddist)

        if eddist > 0:
            # Show Images
            image2 = image[::2, :img_len:2]
            postprocessor.show_all(label, image2, prob2, False)
            image = image.squeeze()
            img255 = as255(1-image)
            prob255 = dilate(as255(1-prob.T), 1, wd_scaled_down_by)
            img = np.vstack((img255, prob255))
            img = Image.fromarray(img)
            img.show()

            input("Want to see one more? Press Enter. Else Ctrl-Z.")

def as255(v):
    return (255*(v-v.min())/(v.max()-v.min())).astype('uint8')


def dilate(a, dh, dw):
    h, w = a.shape                     # H ⨉ W
    a1 = np.stack((a,)*dh, axis=1)     # H ⨉ Dh ⨉ W
    a2 = np.stack((a1,)*dw, axis=-1)   # H ⨉ Dh ⨉ W ⨉ Dw
    return a2.reshape((h*dh, w*dw))    # HDh ⨉ WDw


while True:
    one_batch()
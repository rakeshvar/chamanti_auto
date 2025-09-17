import editdistance
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import tensorflow as tf

from colored import fg, attr
reset = attr('reset')
cols = np.array([f'{fg(231)}█'] + [f'{fg(250 -i)}█' for i in range(19)])

def slab_log(slab, names=None):  # slab is scaled_down_width x chosen_chars
    cslab = cols[np.clip(np.floor(-2*np.log2(slab)).astype(int), 0, 19)]
    if names is None:
        names = ' ' * slab.shape[1]
    for ir, r in enumerate(cslab.T):         # Transposed because chars should be in rows
        print(f'{ir:2d}¦' + ''.join(r) + reset + f'¦ {names[ir]}')


bins1 = [0, .1, .25, .4, .6, .75, .9, 1.0001], '- .*o0@#+'
bins2 = [0, .15, .35, .65, .85, 1.0001],  '- ░▒▓█+'


def get_printer(thresholds, chars):
    def _printer(slab, name=None, col_names=None):
        name and print(f"Name: ", name)
        for irow, row in enumerate(slab):
            print(f'{irow:2d}¦', end='')
            for val in row:
                for t, c in zip(thresholds, chars):
                    if val < t:
                        print(c, end='')
                        break
                else:
                    print(chars[-1], end='')
            print('¦ ', col_names[irow] if col_names else '')
    return _printer


slab_print = get_printer(*bins2)


def charray_to_str(charray):
    return ' '.join(charray)


def myshow(name, labels, chars):
    if len(labels) > 0:
        labels_ = str(np.asarray(labels)).replace('\n', '')
        print(f"{name}: {labels_} {charray_to_str(chars)}")


class PostProcessor:
    def __init__(self, symbols):
        self.n_classes = len(symbols)
        self.symbols = symbols + ['_']

    def labels_to_chars(self, labels):
        return [self.symbols[label] for label in labels]

    def remove_blanks_repeats(self, labels):
        labels_out = []
        for i, label in enumerate(labels):
            if (label != self.n_classes) and (i == 0 or label != labels[i-1]):
                labels_out.append(label)
        return labels_out

    def decode(self, softmax_firings):
        top_labels = np.argmax(softmax_firings, 1)  # (W, A) → (W,)
        return self.remove_blanks_repeats(top_labels)

    def show_all(self, shown_labels, shown_img, softmax_firings, show_imgs):
        shown_chars = self.labels_to_chars(shown_labels)
        myshow('\nShown  ', shown_labels, shown_chars)

        if softmax_firings is not None:
            seen_labels = self.decode(softmax_firings)
            seen_chars = self.labels_to_chars(seen_labels)
            myshow('Seen   ', seen_labels, seen_chars)

            set_seen, set_shown = set(seen_labels), set(shown_labels)

            missed_labels = list(set_shown - set_seen)
            missed_chars = self.labels_to_chars(missed_labels)
            myshow('Missed ', missed_labels, missed_chars)

            extra_labels = list(set_seen - set_shown)
            extra_chars = self.labels_to_chars(extra_labels)
            myshow('Extras ', extra_labels, extra_chars)

            edd = editdistance.eval(shown_labels, seen_labels)/len(shown_labels)
            print(f"Edit Dist: {edd:.1%}")

        if show_imgs:
            print('Image Shown:')
            slab_print(shown_img)

            if softmax_firings is not None:
                print('SoftMax Firings:')
                l = list(shown_labels) + [0, self.n_classes] + extra_labels
                c = shown_chars + ['space', 'blank'] + extra_chars
                slab_log(softmax_firings[:, l], c)

    def editdistances(self, truths, lengths, probabilities, problengths):
        dists = [editdistance.eval(tr[:l], self.decode(pr[:pl]))
                    for pr, pl, tr, l in zip(probabilities, problengths, truths, lengths)]
        return sum(dists) / sum(lengths)

    def show_batch(self,
                   images, image_lengths,        # (B, H, W, 1), (B,)
                   labels, label_lengths,
                   probabilities, probs_lengths, # (B, W, A), (B,)
                   num_samples=5):
        for i in range(num_samples):
            self.show_all(labels[i, :label_lengths[i]],
                          images[i, ::2, :image_lengths[i]:3, 0],  # half rows, third columns (H//2, W'//3)
                          probabilities[i, :probs_lengths[i], :],  # (W//w_down, A)
                          i == num_samples-1)

        return self.editdistances(labels, label_lengths, probabilities, probs_lengths)

    def plot_heatmap(self, probabilities):
        plt.figure(figsize=(12, 6))
        plt.imshow(np.log(probabilities + 1e-10), aspect='auto', cmap='viridis')
        plt.colorbar()
        fig_path = f"/tmp/heatmap{randint(10 ** 5, 10 ** 6)}.png"
        plt.savefig(fig_path)
        print("Saved ", fig_path)
        plt.close()

    def beam_decode(self, logits, input_lengths, beam_width=10, greedy=False):
        """
        Decodes network output into text strings.

        Args:
            logits: model outputs, shape (B, T, num_classes)
            input_lengths: lengths of each sequence (B,) in time steps
            beam_width: beam size for beam search (if greedy=False)
            greedy: if True, use simple argmax decoding

        Returns:
            List of strings (decoded predictions)
        """
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        input_lengths = tf.convert_to_tensor(input_lengths, dtype=tf.int32)

        decoded, _ = tf.keras.backend.ctc_decode(
            logits,
            input_length=input_lengths,
            greedy=greedy,
            beam_width=beam_width
        )
        try:
            decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
        except TypeError as e:
            decoded_dense = decoded[0].numpy()

        ret = []
        for seq in decoded_dense:
            labels = [l for l in seq if l >= 0]
            chars = self.labels_to_chars(labels)
            text = "".join(chars)
            ret.append((labels, chars, text))

        return ret

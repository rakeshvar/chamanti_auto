#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

def dis(arr):
    if arr.ndim == 4:
        arr  = np.vstack(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = (255*(1-arr)).astype(np.uint8)
    display(im.fromarray(arr))

def cdf(a):
    plt.plot(np.sort(a), np.arange(len(a))/len(a))


# In[2]:


import telugu as lang
from Lekhaka import Scribe, DataGenerator, Noiser


# In[3]:


ht = 48
cps = 12
bsz = 8


# In[4]:


scribe_args = {'height': ht, 'hbuffer': 5, 'vbuffer': 0, 'nchars_per_sample': cps}
scriber = Scribe(lang, **scribe_args)
wd = scriber.width
noiser=Noiser(scriber.width//8, .92, 1, ht//8)
datagen = DataGenerator(scriber, noiser=noiser, batch_size=bsz)


# In[12]:


images, labels, image_lengths, label_lengths = datagen.get()
dis(images)


# In[13]:


get_ipython().run_line_magic('autoreload', '2')
import deformer as dfa

deformer_args = {
    "zoom_range":.1,
    "translation_range":(0.1, .05),
    "shear_range":.01,
    "warp_intensity": .03,
    "ink_blot_fraction": .1,
    "erosion_intensity":.08,
    "binarize_threshold": .1,
    "jpeg_quality_range":(20, 80),
    "brightness_range":.2,
    "contrast_range":.3,
}
deformer = dfa.Deformer(**deformer_args)
deformed = deformer(images, training=True)
dis(np.concatenate((images, deformed), axis=2))


# In[29]:


images, labels, image_lengths, label_lengths = datagen.get()
deformed = deformer(images, training=True)
dis(np.concatenate((images, deformed), axis=2))


# In[23]:


deformer.get_config()


# In[ ]:





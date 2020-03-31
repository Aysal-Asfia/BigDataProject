#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.io import loadmat

labels = pd.read_pickle("ROIs/BOLD5000_imagenet_superlabels.pkl")
keys_to_remove = ['__header__', '__version__', '__globals__']


def process_data(mat_files, image_name_file):
    CSI1_TR1 = loadmat(mat_files[0])
    CSI1_TR2 = loadmat(mat_files[1])
    CSI1_TR3 = loadmat(mat_files[2])
    CSI1_TR4 = loadmat(mat_files[3])
    CSI1_TR5 = loadmat(mat_files[4])
    CSI1_TR34 = loadmat(mat_files[5])

    images = open(image_name_file, "r")
    d_images = {"image": np.array([line.strip('\n') for line in images])}

    [CSI1_TR1.pop(key) for key in keys_to_remove]
    CSI1_TR1 = {k + '_1': v for k, v in CSI1_TR1.items()}

    [CSI1_TR2.pop(key) for key in keys_to_remove]
    CSI1_TR2 = {k + '_2': v for k, v in CSI1_TR2.items()}

    [CSI1_TR3.pop(key) for key in keys_to_remove]
    CSI1_TR3 = {k + '_3': v for k, v in CSI1_TR3.items()}

    [CSI1_TR4.pop(key) for key in keys_to_remove]
    CSI1_TR4 = {k + '_4': v for k, v in CSI1_TR4.items()}

    [CSI1_TR5.pop(key) for key in keys_to_remove]
    CSI1_TR5 = {k + '_5': v for k, v in CSI1_TR5.items()}

    [CSI1_TR34.pop(key) for key in keys_to_remove]
    CSI1_TR34 = {k + '_34': v for k, v in CSI1_TR34.items()}

    image_list = d_images["image"]
    image_food = labels["food"]
    image_living_inanimate = labels["living inanimate"]
    image_living_animate = labels["living animate"]
    image_objects = labels["objects"]
    image_geo = labels["geo"]

    label_list = []
    label_dict = {}

    for image in image_list:
        if image in image_food:
            label_list.append("food")
        elif image in image_living_inanimate:
            label_list.append("living inanimate")
        elif image in image_living_animate:
            label_list.append("living animate")
        elif image in image_objects:
            label_list.append("objects")
        elif image in image_geo:
            label_list.append("geo")
        else:
            label_list.append("0")

    label_dict["label"] = np.array(label_list)

    d_concat = {**CSI1_TR1, **CSI1_TR2, **CSI1_TR3, **CSI1_TR4, **CSI1_TR5, **CSI1_TR34, **d_images, **label_dict}

    index_pos_list = [i for i in range(len(label_list)) if label_list[i] == '0']
    final_concat = {}
    for key in d_concat.keys():
        old_value_np = d_concat[key]
        new_value_np = np.delete(old_value_np, index_pos_list, axis=0)
        final_concat[key] = new_value_np

    return final_concat


data = []
for i in range(4):
    index = i + 1
    mat_files = ['ROIs/CSI%d/mat/CSI%d_ROIs_TR1.mat' % (index, index),
                 'ROIs/CSI%d/mat/CSI%d_ROIs_TR2.mat' % (index, index),
                 'ROIs/CSI%d/mat/CSI%d_ROIs_TR3.mat' % (index, index),
                 'ROIs/CSI%d/mat/CSI%d_ROIs_TR4.mat' % (index, index),
                 'ROIs/CSI%d/mat/CSI%d_ROIs_TR5.mat' % (index, index),
                 'ROIs/CSI%d/mat/CSI%d_ROIs_TR34.mat' % (index, index)]
    image_name_file = "ROIs/stim_lists/CSI0%d_stim_lists.txt" % index
    data.append(process_data(mat_files, image_name_file))

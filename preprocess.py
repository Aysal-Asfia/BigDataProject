import numpy as np
import pandas as pd
from scipy.io import loadmat

image_label_file = "ROIs/BOLD5000_imagenet_superlabels.pkl"
keys_to_remove = ['__header__', '__version__', '__globals__']


def final_concat(CSI_TR, index_pos_list):
    values = [np.delete(CSI_TR[key], index_pos_list, axis=0) for key in CSI_TR.keys()]
    return np.hstack(values)


def process_data(section_number, no_steps=5):
    steps = 6 if no_steps == 6 else 5

    CSI_TR1 = loadmat('ROIs/CSI%d/mat/CSI%d_ROIs_TR1.mat' % (section_number, section_number))
    CSI_TR2 = loadmat('ROIs/CSI%d/mat/CSI%d_ROIs_TR2.mat' % (section_number, section_number))
    CSI_TR3 = loadmat('ROIs/CSI%d/mat/CSI%d_ROIs_TR3.mat' % (section_number, section_number))
    CSI_TR4 = loadmat('ROIs/CSI%d/mat/CSI%d_ROIs_TR4.mat' % (section_number, section_number))
    CSI_TR5 = loadmat('ROIs/CSI%d/mat/CSI%d_ROIs_TR5.mat' % (section_number, section_number))
    CSI_TR34 = loadmat('ROIs/CSI%d/mat/CSI%d_ROIs_TR34.mat' % (section_number, section_number))

    images = open("ROIs/stim_lists/CSI0%d_stim_lists.txt" % section_number, "r")
    d_images = {"image": np.array([[line.strip('\n')] for line in images])}
    labels = pd.read_pickle(image_label_file)

    CSI1_TR_list = [CSI_TR1, CSI_TR2, CSI_TR3, CSI_TR4, CSI_TR5, CSI_TR34][:steps]
    # new_CSI1_TR_list = []

    for TR in CSI1_TR_list:
        for key in keys_to_remove:
            TR.pop(key)
        # new_CSI1_TR_list.append(TR)

    label_list = []
    for image in d_images["image"]:
        if image in labels["food"]:
            label_list.append(["food"])
        elif image in labels["living inanimate"]:
            label_list.append(["living inanimate"])
        elif image in labels["living animate"]:
            label_list.append(["living animate"])
        elif image in labels["objects"]:
            label_list.append(["objects"])
        elif image in labels["geo"]:
            label_list.append(["geo"])
        else:
            label_list.append(["0"])

    y = np.array([label for label in label_list if label[0] != '0'])
    index_pos_list = [i for i in range(len(label_list)) if label_list[i][0] == '0']
    x = np.rollaxis(np.dstack([final_concat(item, index_pos_list) for item in CSI1_TR_list]), 2, 1)

    return x, y

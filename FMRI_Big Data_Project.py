#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from scipy.io import savemat

CSI1_TR1 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI1/mat/CSI1_ROIs_TR1.mat')
CSI1_TR2 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI1/mat/CSI1_ROIs_TR2.mat')
CSI1_TR3 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI1/mat/CSI1_ROIs_TR3.mat')
CSI1_TR4 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI1/mat/CSI1_ROIs_TR4.mat')
CSI1_TR5 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI1/mat/CSI1_ROIs_TR5.mat')
CSI1_TR34 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI1/mat/CSI1_ROIs_TR34.mat')

Labels = pd.read_pickle("C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/BOLD5000_imagenet_superlabels.pkl")

Images_1 = open("C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/stim_lists/CSI01_stim_lists.txt", "r")
d_Images_1 = {"Image" : np.array([line.strip('\n') for line in Images_1])}

keys_to_remove = ['__header__', '__version__', '__globals__']

############ Section 1 #################

[CSI1_TR1.pop(key) for key in keys_to_remove]
CSI1_TR1 = {k+'_1': v for k, v in CSI1_TR1.items()}

[CSI1_TR2.pop(key) for key in keys_to_remove]
CSI1_TR2 = {k+'_2': v for k, v in CSI1_TR2.items()}

[CSI1_TR3.pop(key) for key in keys_to_remove]
CSI1_TR3 = {k+'_3': v for k, v in CSI1_TR3.items()}

[CSI1_TR4.pop(key) for key in keys_to_remove]
CSI1_TR4 = {k+'_4': v for k, v in CSI1_TR4.items()}

[CSI1_TR5.pop(key) for key in keys_to_remove]
CSI1_TR5 = {k+'_5': v for k, v in CSI1_TR5.items()}

[CSI1_TR34.pop(key) for key in keys_to_remove]
CSI1_TR34 = {k+'_34': v for k, v in CSI1_TR34.items()}

image_list_1 = d_Images_1["Image"]
Image_food_1 = Labels["food"]
Image_living_inanimate_1 = Labels["living inanimate"]
Image_living_animate_1 = Labels["living animate"]
Image_objects_1 = Labels["objects"]
Image_geo_1 = Labels["geo"]

label_list_1 = []
label_dict_1 = {}

for image in image_list_1:
    if image in Image_food_1:
        label_list_1.append("food")
    elif image in Image_living_inanimate_1:
        label_list_1.append("living inanimate")
    elif image in Image_living_animate_1:
        label_list_1.append("living animate")
    elif image in Image_objects_1:
        label_list_1.append("objects")
    elif image in Image_geo_1:
        label_list_1.append("geo")
    else:
        label_list_1.append("0")

label_dict_1["Label"] = np.array(label_list_1)

   
d_concat_1 = {**CSI1_TR1, **CSI1_TR2, **CSI1_TR3, **CSI1_TR4, **CSI1_TR5, **CSI1_TR34, **d_Images_1, **label_dict_1}

indexPosList_1 = [ i for i in range(len(label_list_1)) if label_list_1[i] == '0' ]
final_concat_1 = {}
for key in d_concat_1.keys():
    old_value_np_1 = d_concat_1[key]
    new_value_np_1 = np.delete(old_value_np_1, indexPosList_1, axis=0)
    final_concat_1[key] = new_value_np_1

    
############ Section 2 #################

CSI2_TR1 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI2/mat/CSI2_ROIs_TR1.mat')
CSI2_TR2 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI2/mat/CSI2_ROIs_TR2.mat')
CSI2_TR3 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI2/mat/CSI2_ROIs_TR3.mat')
CSI2_TR4 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI2/mat/CSI2_ROIs_TR4.mat')
CSI2_TR5 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI2/mat/CSI2_ROIs_TR5.mat')
CSI2_TR34 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI2/mat/CSI2_ROIs_TR34.mat')

Images_2 = open("C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/stim_lists/CSI02_stim_lists.txt", "r")
d_Images_2 = {"Image" : np.array([line.strip('\n') for line in Images_2])}

[CSI2_TR1.pop(key) for key in keys_to_remove]
CSI2_TR1 = {k+'_1': v for k, v in CSI2_TR1.items()}

[CSI2_TR2.pop(key) for key in keys_to_remove]
CSI2_TR2 = {k+'_2': v for k, v in CSI2_TR2.items()}

[CSI2_TR3.pop(key) for key in keys_to_remove]
CSI2_TR3 = {k+'_3': v for k, v in CSI2_TR3.items()}

[CSI2_TR4.pop(key) for key in keys_to_remove]
CSI2_TR4 = {k+'_4': v for k, v in CSI2_TR4.items()}

[CSI2_TR5.pop(key) for key in keys_to_remove]
CSI2_TR5 = {k+'_5': v for k, v in CSI2_TR5.items()}

[CSI2_TR34.pop(key) for key in keys_to_remove]
CSI2_TR34 = {k+'_34': v for k, v in CSI2_TR34.items()}

image_list_2 = d_Images_2["Image"]
Image_food_2 = Labels["food"]
Image_living_inanimate_2 = Labels["living inanimate"]
Image_living_animate_2 = Labels["living animate"]
Image_objects_2 = Labels["objects"]
Image_geo_2 = Labels["geo"]

label_list_2 = []
label_dict_2 = {}

for image in image_list_2:
    if image in Image_food_2:
        label_list_2.append("food")
    elif image in Image_living_inanimate_2:
        label_list_2.append("living inanimate")
    elif image in Image_living_animate_2:
        label_list_2.append("living animate")
    elif image in Image_objects_2:
        label_list_2.append("objects")
    elif image in Image_geo_2:
        label_list_2.append("geo")
    else:
        label_list_2.append("0")

label_dict_2["Label"] = np.array(label_list_2)

   
d_concat_2 = {**CSI2_TR1, **CSI2_TR2, **CSI2_TR3, **CSI2_TR4, **CSI2_TR5, **CSI2_TR34, **d_Images_2, **label_dict_2}

indexPosList_2 = [ i for i in range(len(label_list_2)) if label_list_2[i] == '0' ]
final_concat_2 = {}
for key in d_concat_2.keys():
    old_value_np_2 = d_concat_2[key]
    new_value_np_2 = np.delete(old_value_np_2, indexPosList_2, axis=0)
    final_concat_2[key] = new_value_np_2

    
############ Section 3 #################    

CSI3_TR1 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI3/mat/CSI3_ROIs_TR1.mat')
CSI3_TR2 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI3/mat/CSI3_ROIs_TR2.mat')
CSI3_TR3 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI3/mat/CSI3_ROIs_TR3.mat')
CSI3_TR4 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI3/mat/CSI3_ROIs_TR4.mat')
CSI3_TR5 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI3/mat/CSI3_ROIs_TR5.mat')
CSI3_TR34 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI3/mat/CSI3_ROIs_TR34.mat')

Images_3 = open("C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/stim_lists/CSI03_stim_lists.txt", "r")
d_Images_3 = {"Image" : np.array([line.strip('\n') for line in Images_3])}

[CSI3_TR1.pop(key) for key in keys_to_remove]
CSI3_TR1 = {k+'_1': v for k, v in CSI3_TR1.items()}

[CSI3_TR2.pop(key) for key in keys_to_remove]
CSI3_TR2 = {k+'_2': v for k, v in CSI3_TR2.items()}

[CSI3_TR3.pop(key) for key in keys_to_remove]
CSI3_TR3 = {k+'_3': v for k, v in CSI3_TR3.items()}

[CSI3_TR4.pop(key) for key in keys_to_remove]
CSI3_TR4 = {k+'_4': v for k, v in CSI3_TR4.items()}

[CSI3_TR5.pop(key) for key in keys_to_remove]
CSI3_TR5 = {k+'_5': v for k, v in CSI3_TR5.items()}

[CSI3_TR34.pop(key) for key in keys_to_remove]
CSI3_TR34 = {k+'_34': v for k, v in CSI3_TR34.items()}

image_list_3 = d_Images_3["Image"]
Image_food_3 = Labels["food"]
Image_living_inanimate_3 = Labels["living inanimate"]
Image_living_animate_3 = Labels["living animate"]
Image_objects_3 = Labels["objects"]
Image_geo_3 = Labels["geo"]

label_list_3 = []
label_dict_3 = {}

for image in image_list_3:
    if image in Image_food_3:
        label_list_3.append("food")
    elif image in Image_living_inanimate_3:
        label_list_3.append("living inanimate")
    elif image in Image_living_animate_3:
        label_list_3.append("living animate")
    elif image in Image_objects_3:
        label_list_3.append("objects")
    elif image in Image_geo_3:
        label_list_3.append("geo")
    else:
        label_list_3.append("0")

label_dict_3["Label"] = np.array(label_list_3)

   
d_concat_3 = {**CSI3_TR1, **CSI3_TR2, **CSI3_TR3, **CSI3_TR4, **CSI3_TR5, **CSI3_TR34, **d_Images_3, **label_dict_3}

indexPosList_3 = [ i for i in range(len(label_list_3)) if label_list_3[i] == '0' ]
final_concat_3 = {}
for key in d_concat_3.keys():
    old_value_np_3 = d_concat_3[key]
    new_value_np_3 = np.delete(old_value_np_3, indexPosList_3, axis=0)
    final_concat_3[key] = new_value_np_3


############ Section 4 #################

CSI4_TR1 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI4/mat/CSI4_ROIs_TR1.mat')
CSI4_TR2 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI4/mat/CSI4_ROIs_TR2.mat')
CSI4_TR3 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI4/mat/CSI4_ROIs_TR3.mat')
CSI4_TR4 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI4/mat/CSI4_ROIs_TR4.mat')
CSI4_TR5 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI4/mat/CSI4_ROIs_TR5.mat')
CSI4_TR34 = loadmat('C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/CSI4/mat/CSI4_ROIs_TR34.mat')

Images_4 = open("C:/Users/umroot/Desktop/BOLD5000/New folder/ROIs/stim_lists/CSI04_stim_lists.txt", "r")
d_Images_4 = {"Image" : np.array([line.strip('\n') for line in Images_4])}

[CSI4_TR1.pop(key) for key in keys_to_remove]
CSI4_TR1 = {k+'_1': v for k, v in CSI4_TR1.items()}

[CSI4_TR2.pop(key) for key in keys_to_remove]
CSI4_TR2 = {k+'_2': v for k, v in CSI4_TR2.items()}

[CSI4_TR3.pop(key) for key in keys_to_remove]
CSI4_TR3 = {k+'_3': v for k, v in CSI4_TR3.items()}

[CSI4_TR4.pop(key) for key in keys_to_remove]
CSI4_TR4 = {k+'_4': v for k, v in CSI4_TR4.items()}

[CSI4_TR5.pop(key) for key in keys_to_remove]
CSI4_TR5 = {k+'_5': v for k, v in CSI4_TR5.items()}

[CSI4_TR34.pop(key) for key in keys_to_remove]
CSI4_TR34 = {k+'_34': v for k, v in CSI4_TR34.items()}

image_list_4 = d_Images_4["Image"]
Image_food_4 = Labels["food"]
Image_living_inanimate_4 = Labels["living inanimate"]
Image_living_animate_4 = Labels["living animate"]
Image_objects_4 = Labels["objects"]
Image_geo_4 = Labels["geo"]

label_list_4 = []
label_dict_4 = {}

for image in image_list_4:
    if image in Image_food_4:
        label_list_4.append("food")
    elif image in Image_living_inanimate_4:
        label_list_4.append("living inanimate")
    elif image in Image_living_animate_4:
        label_list_4.append("living animate")
    elif image in Image_objects_4:
        label_list_4.append("objects")
    elif image in Image_geo_4:
        label_list_4.append("geo")
    else:
        label_list_4.append("0")

label_dict_4["Label"] = np.array(label_list_4)

   
d_concat_4 = {**CSI4_TR1, **CSI4_TR2, **CSI4_TR3, **CSI4_TR4, **CSI4_TR5, **CSI4_TR34, **d_Images_4, **label_dict_4}

indexPosList_4 = [ i for i in range(len(label_list_4)) if label_list_4[i] == '0' ]
final_concat_4 = {}
for key in d_concat_4.keys():
    old_value_np_4 = d_concat_4[key]
    new_value_np_4 = np.delete(old_value_np_4, indexPosList_4, axis=0)
    final_concat_4[key] = new_value_np_4


final_concat_1
final_concat_2
final_concat_3
final_concat_4


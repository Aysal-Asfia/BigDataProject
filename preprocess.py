from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import os.path as op
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

########################################################################
def data_proces(data_dir):
    Labels = pd.read_pickle(data_dir + '/BOLD5000_imagenet_superlabels.pkl')
    Image_food = Labels["food"]
    Image_living_inanimate = Labels["living inanimate"]
    Image_living_animate = Labels["living animate"]
    Image_objects = Labels["objects"]
    Image_geo = Labels["geo"]

    keys_to_remove = ['__header__', '__version__', '__globals__']

    label_list = []
    def label_image(image):
        if image in Image_food:
            label_list.append(["food"])
        elif image in Image_living_inanimate:
            label_list.append(["living inanimate"])
        elif image in Image_living_animate:
            label_list.append(["living animate"])
        elif image in Image_objects:
            label_list.append(["objects"])
        elif image in Image_geo:
            label_list.append(["geo"])
        else:
            label_list.append(["0"])
        return label_list


    def remove(old_value_np, indexPosList):
        new_value_np = np.delete(old_value_np, indexPosList, axis=0)
        return new_value_np


    def final_cancat(CSI_TR, c, indexPosList):
        final_concat = {}
        for key in CSI_TR.keys():
            final_concat[key]  = remove(CSI_TR[key], indexPosList)

        a = np.hstack(list(final_concat.values()))
        return a

    ######################   Section 1   #######################
    CSI1_TR1 = loadmat(data_dir + '/CSI1/mat/CSI1_ROIs_TR1.mat')
    CSI1_TR2 = loadmat(data_dir + '/CSI1/mat/CSI1_ROIs_TR2.mat')
    CSI1_TR3 = loadmat(data_dir + '/CSI1/mat/CSI1_ROIs_TR3.mat')
    CSI1_TR4 = loadmat(data_dir + '/CSI1/mat/CSI1_ROIs_TR4.mat')
    CSI1_TR5 = loadmat(data_dir + '/CSI1/mat/CSI1_ROIs_TR5.mat')
    CSI1_TR34 = loadmat(data_dir + '/CSI1/mat/CSI1_ROIs_TR34.mat')

    Images_1 = open(data_dir + '/stim_lists/CSI01_stim_lists.txt', "r")
    d_Images_1 = {"Image" : np.array([[line.strip('\n')] for line in Images_1])}


    CSI1_TR_list = [CSI1_TR1, CSI1_TR2, CSI1_TR3, CSI1_TR4, CSI1_TR5, CSI1_TR34]
    new_CSI1_TR_list = []
    for TR in CSI1_TR_list:
        [TR.pop(key) for key in keys_to_remove]
        new_CSI1_TR_list.append(TR)

    for image in d_Images_1["Image"]:
        label_list_1 = label_image(image)

    label_list_1_new = []
    [label_list_1_new.append(label) for label in label_list_1 if label[0] != '0']
    label_array_1 = np.array(label_list_1_new)
    
    indexPosList_1 = [ i for i in range(len(label_list_1)) if label_list_1[i][0] == '0' ]

    c_1 = 1
    concat_list_1 = []
    for item in new_CSI1_TR_list:
        concat_list_1.append(final_cancat(item, c_1, indexPosList_1))
        c_1 = c_1 + 1

    Xx1_1 = np.hstack((i for i in concat_list_1))
    Xx2_1 = np.dstack([i for i in concat_list_1])

    ######################   Section 2   #######################
    CSI2_TR1 = loadmat(data_dir + '/CSI2/mat/CSI2_ROIs_TR1.mat')
    CSI2_TR2 = loadmat(data_dir + '/CSI2/mat/CSI2_ROIs_TR2.mat')
    CSI2_TR3 = loadmat(data_dir + '/CSI2/mat/CSI2_ROIs_TR3.mat')
    CSI2_TR4 = loadmat(data_dir + '/CSI2/mat/CSI2_ROIs_TR4.mat')
    CSI2_TR5 = loadmat(data_dir + '/CSI2/mat/CSI2_ROIs_TR5.mat')
    CSI2_TR34 = loadmat(data_dir + '/CSI2/mat/CSI2_ROIs_TR34.mat')

    Images_2 = open(data_dir + '/stim_lists/CSI02_stim_lists.txt', "r")
    d_Images_2 = {"Image" : np.array([[line.strip('\n')] for line in Images_2])}

    CSI2_TR_list = [CSI2_TR1, CSI2_TR2, CSI2_TR3, CSI2_TR4, CSI2_TR5, CSI2_TR34]
    new_CSI2_TR_list = []
    for TR in CSI2_TR_list:
        [TR.pop(key) for key in keys_to_remove]
        new_CSI2_TR_list.append(TR)

    for image in d_Images_2["Image"]:
        label_list_2 = label_image(image)

    label_list_2_new = []
    [label_list_2_new.append(label) for label in label_list_2 if label[0] != '0']
    label_array_2 = np.array(label_list_2_new)
    
    indexPosList_2 = [ i for i in range(len(label_list_2)) if label_list_2[i][0] == '0' ]

    c_2 = 1
    concat_list_2 = []
    for item in new_CSI1_TR_list:
        concat_list_2.append(final_cancat(item, c_2, indexPosList_2))
        c_2 = c_2 + 1

    Xx1_2 = np.hstack((i for i in concat_list_2))
    Xx2_2 = np.dstack([i for i in concat_list_2])

    ######################   Section 3   #######################
    CSI3_TR1 = loadmat(data_dir + '/CSI3/mat/CSI3_ROIs_TR1.mat')
    CSI3_TR2 = loadmat(data_dir + '/CSI3/mat/CSI3_ROIs_TR2.mat')
    CSI3_TR3 = loadmat(data_dir + '/CSI3/mat/CSI3_ROIs_TR3.mat')
    CSI3_TR4 = loadmat(data_dir + '/CSI3/mat/CSI3_ROIs_TR4.mat')
    CSI3_TR5 = loadmat(data_dir + '/CSI3/mat/CSI3_ROIs_TR5.mat')
    CSI3_TR34 = loadmat(data_dir + '/CSI3/mat/CSI3_ROIs_TR34.mat')

    Images_3 = open(data_dir + '/stim_lists/CSI03_stim_lists.txt', "r")
    d_Images_3 = {"Image" : np.array([[line.strip('\n')] for line in Images_3])}

    CSI3_TR_list = [CSI3_TR1, CSI3_TR2, CSI3_TR3, CSI3_TR4, CSI3_TR5, CSI3_TR34]
    new_CSI3_TR_list = []
    for TR in CSI3_TR_list:
        [TR.pop(key) for key in keys_to_remove]
        new_CSI3_TR_list.append(TR)

    for image in d_Images_3["Image"]:
        label_list_3 = label_image(image)

    label_list_3_new = []
    [label_list_3_new.append(label) for label in label_list_3 if label[0] != '0']
    label_array_3 = np.array(label_list_3_new)
    
    indexPosList_3 = [ i for i in range(len(label_list_3)) if label_list_3[i][0] == '0' ]

    c_3 = 1
    concat_list_3 = []
    for item in new_CSI3_TR_list:
        concat_list_3.append(final_cancat(item, c_3, indexPosList_3))
        c_3 = c_3 + 1

    Xx1_3 = np.hstack((i for i in concat_list_3))
    Xx2_3 = np.dstack([i for i in concat_list_3])

    ######################   Section 4   #######################
    CSI4_TR1 = loadmat(data_dir + '/CSI4/mat/CSI4_ROIs_TR1.mat')
    CSI4_TR2 = loadmat(data_dir + '/CSI4/mat/CSI4_ROIs_TR2.mat')
    CSI4_TR3 = loadmat(data_dir + '/CSI4/mat/CSI4_ROIs_TR3.mat')
    CSI4_TR4 = loadmat(data_dir + '/CSI4/mat/CSI4_ROIs_TR4.mat')
    CSI4_TR5 = loadmat(data_dir + '/CSI4/mat/CSI4_ROIs_TR5.mat')
    CSI4_TR34 = loadmat(data_dir + '/CSI4/mat/CSI4_ROIs_TR34.mat')

    Images_4 = open(data_dir + '/stim_lists/CSI04_stim_lists.txt', "r")
    d_Images_4 = {"Image" : np.array([[line.strip('\n')] for line in Images_4])}

    CSI4_TR_list = [CSI4_TR1, CSI4_TR2, CSI4_TR3, CSI4_TR4, CSI4_TR5, CSI4_TR34]
    new_CSI4_TR_list = []
    for TR in CSI4_TR_list:
        [TR.pop(key) for key in keys_to_remove]
        new_CSI4_TR_list.append(TR)

    for image in d_Images_4["Image"]:
        label_list_4 = label_image(image)

    label_list_4_new = []
    [label_list_4_new.append(label) for label in label_list_4 if label[0] != '0']
    label_array_4 = np.array(label_list_4_new)
    
    indexPosList_4 = [ i for i in range(len(label_list_4)) if label_list_4[i][0] == '0' ]

    c_4 = 1
    concat_list_4 = []
    for item in new_CSI4_TR_list:
        concat_list_4.append(final_cancat(item, c_4, indexPosList_4))
        c_4 = c_4 + 1

    Xx1_4 = np.hstack((i for i in concat_list_4))
    Xx2_4 = np.dstack([i for i in concat_list_4]) 
    
    
    return concat_list_1, label_array_1, Xx1_1, Xx2_1,\
concat_list_2, label_array_2, Xx1_1, Xx2_2,\
concat_list_3, label_array_3, Xx1_3, Xx2_3,\
concat_list_4, label_array_4, Xx1_4, Xx2_4

############################ Main ##############################
def main():
    data_dir = input("Please enter direction of your data files: ")
    return data_proces(data_dir)
#     print("++++++++++++++++++  Section 1  +++++++++++++++++++++++++++++++++")
#     print(Xx1_1.shape)
#     print(Xx2_1.shape)
    
#     print("++++++++++++++++++  Section 2  +++++++++++++++++++++++++++++++++")
#     print(Xx1_2.shape)
#     print(Xx2_2.shape)
    
#     print("++++++++++++++++++  Section 3  +++++++++++++++++++++++++++++++++")
#     print(Xx1_3.shape)
#     print(Xx2_3.shape)
    
#     print("++++++++++++++++++  Section 4  +++++++++++++++++++++++++++++++++")
#     print(Xx1_4.shape)
#     print(Xx2_4.shape)
    

main()

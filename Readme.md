# Observed objects classification based on fMRI data

## **Abstract** 

Functional magnetic resonance imaging (fMRI) is one of the most widely used tools to  study the  neural underpinnings of  
human cognition. In this  project,we apply deep learning algorithms to classify objects observed by subjects 
using fMRI data. Subjects involved in the experiments were shown different images of static objects as well as 
animated images. Meanwhile, in the field of neuroimaging,  increasing spatial and temporal resolutions as well as 
larger sample sizes lead to a rapid increase in the amount of data that needs to be processed in a typical study.  
That is explosion of big imaging data highlights new challenges,such as data sharing, management, and processing, 
as well as reproducibility. In the work, we address the big data management challenges for the problem of fMRI images 
classification.  We profile the performance of the algorithm on used data set in terms of used resources such as 
CPU and memory in different steps, from data pre-processing to model training, validation and testing.


## **I. Introduction**

Neuroimaging is a relatively new field emerged with the development in neuroscience, psychology and computer science 
which works with brain images to capture and analyze the structures and functions in parts of neural system. 
Nowadays, due to the excellent resolution of fMRI data, the images containing visual objects can be effectively 
recognized using specific patterns derived from this data that are recorded when images are displayed on screen in 
front of the subjects [5].  In this project, we apply deep learning algorithms to classify  objects  observed  by  
subjects using fMRI data. Subjects involved in the experiments were shown different images of static objects as well as 
animated images.  The fMRI data of their brains when they were looking at the images was recorded in a few time steps.  
This is the input for our classification algorithm. 

Meanwhile, nowadays in the field of neuroimaging, with the augmentation of fMRI datasets, increasing spatial and 
temporal resolutions as well as larger sample sizes lead to a rapid increase in the amount of data that needs to be 
processed in a typical study [1]. For more clarification, based on paper [3], by 2015 the amount of acquired 
neuroimaging data alone, discounting header informa-tion and before more files are generated during data processing and 
statistical analysis, exceeded an average of 20GB per published research study.  However,as the richness of brain 
datasets continues to grow and the push to place it inaccessible repositories mounts, there are many issues to be 
considered on how to handle the data, move it from place to place, how to store it, analyze it, and share it [3].  
Because of that, in paper [3], the authors indicated it is safe to say that human neuroimaging is now, officially, 
a “big data” science.

Furthermore, in neuroimaging field in many applications, the problems that neuroscientists as well as 
computer scientists are tackling with are not only which algorithms to apply, what is the best configuration 
for the algorithms, but also the problems in resource optimization, reproducibility and fault-tolerance. 
Thus, in this project, we profile the performance of the applied deep-learning technique on the applied dataset 
in terms of used resources such as CPU and memory in different steps, from data pre-processing to model training, 
validation and testing.  By doing this analysis, our expectation is to have an idea of where could be the bottleneck, 
what could be improved to optimize resource usage.

## **II. Materials and methods**. 

### **1.  Materials**
    
### **1.1 Dataset**

For  this  work,  we  used  BOLD5000  repository  [2],  a  human  functional  MRI(fMRI) study. This repository includes 
almost 5,000 distinct images depicting real-world scenes as brain stimuli. However, from this repository we focused only 
on ImageNet dataset, providing super-categories labels.  In this fMRI dataset, 4 participants were scanned in a 
slow-evented related design with 4,916 unique scenes. 

The fMRI data was collected over 16 sessions, 15 of which were task-related sessions, plus an additional session for 
acquiring high resolution anatomical scans. Images were presented 
for 1 second, with 9 seconds of fixation between trials.  Participants were asked to judge whether they liked, disliked, 
or were neutral about the images.  As aforementioned, the images in ImageNet dataset are labeled with ImageNet 
super categories. These super categories were created by using the WordNet hierarchy. From 61 final WordNet categories, 
images are labeled as ”Living Animate”, “Living Inanimate”, “Objects”, “Food” or “Geography”. An example of label 
mapping is “Dog” to “Living Animate” and“Vehicle” to “Objects”. 

In this work, features of fMRI data were represented 
by using ROI regions including: LHPPA’, ’RHLOC’, ’LHLOC’, ’RHEarlyVis’, ’RHRSC’, ’RHOPA’,’RHPPA’, ’LHEarlyVis’, ’LHRSC’, 
’LHOPA’. The data of the ROIs were ex-tracted across the time-course of the trial presentation 
(TR1 = 0−2s,TR2 =32−4s,TR3=4−6s,TR4=6−8s,TR5=8−10s). In this dataset, for the first subject, the number of experiments 
and features were 1916 and 1685, respectively. We have the same number of experiments for the other subjects. 
However, the number of features for other three subjects were 2270, 3104, 2787 respectively.

### **1.2 Technologies**

In this project, we use **_tensorflow_**, **_keras_**, which is widely used for deep learning,
and **_sklearn_** for data preprocessing and performance evaluation. 
 

### **1.3 LSTM classification**

For the problem of fMRI images classification, type of times series based data classification, we choose 
Long Short Term Memory (LSTM) as our classifier. LSTM is based on recurrent neural network (RNN), which is a 
deep learning algorithm. In the following, we present a brief review on LSTM networks. 

The main motivation of applying 
RNNs is extracting higher-dimensional dependencies from sequential data. The main characteristic of RNN units is that 
they have connections not only between the subsequent layers,but also among themselves to capture information from 
previous inputs. Even though traditional RNNs can easily learn short-term dependencies; however,they have difficulties 
in learning long-term dynamics due to the vanishing and exploding gradient problems. The main advantage of LSTM is 
addressing the vanishing and exploding gradient problems by learning both long- and short-term dependencies [4].

**_I am not sure if we should describe LSTM cell with details or we should remove it*******_** 
An LSTM network is composed of cells, whose outputsevolvethrough the network based on past memory content. 
The cells have a common cell state, keeping long-term dependencies along the entire LSTM chain of cells. The following 
information is then controlled by the input gate (it) and forgetgate (ft), thus allowing the network to decide whether 
to forget the previousstate (Ct1) or update the current state(Ct) with new information. The output2 of each cell 
(hidden state) is controlled by an output gate (ot), allowing the cell to compute its output given the updated 
cell state [4].  The formulas describing an LSTM cell architecture are presented as:

_it=σ(Wi.[ht−1,xt] +bi(1)

ft=σ(Wf.[ht−1,xt] +bf(2)

Ct=ft∗Ct−1+it∗tanh(Wc.[ht−1,xt] +bc) (3)

ot=σ(Wo.[ht−1,xt] +bo(4)

ht=ot∗tanh(Ct) (5)_

In our work, the proposed network includes five LSTM layers and one dense layer. In all of the layers, we used tanh 
activation function. But we applied softmax activation function in the last dense layer to predict the probability 
of each class. We set the dropout rate as 0.2 in all LSTM layers.

### **2. Method**

For this classification problem, since the fMRI images are in time series and are related,
we choose LSTM as our classifier. 
The model is trained with batch size of 50 and Adam as the optimizer. 
After the model is trained, we use some metrics to measure the performance of out model. Apart from accuracy, 
we will use ROC curve and confusion matrix to evaluate since this is a multiclass classification problem and 
the classes are imbalance.

Since we want to analyze the resource usage when the model is trained, we do not pre-process the data 
and train our model with the original data. By doing this, the training will be compute-intensive as well as 
memory consuming, and the impact of model and hyperparameter choices can be more visible. This may make the bottlenecks 
to be identified and the improved more easily.

When it comes to resource profiling, we focus on CPU time, memory usage and cache used if possible as these usually 
are the possible bottlenecks to be optimized.

### **3. Result**

The model is trained with batch size of 50 applying Adam optimizer. After the model is trained, we use some metrics 
to measure the performance of our model. Table 1 shows the performance of model for test dataset in term of several 
metrics: precision, recall, f1-score and support. Our results show that the total accuracy equals to 0.7318. 
Furthermore, Table 2 illustrates the corresponding confusion matrix for the trained model on the test dataset.

_we will use ROC curve and confusion matrix to evaluate since this is a multi-class classification problem and 
the classes are imbalance._

_**Table 1. Classification measurements report**_

| index | precision | recall | f1_score | support |
|-------|-----------|--------|----------|---------|
| 1     | 0.33      | 0.10   | 0.15     | 20      |
| 2     | 0.00      | 0.00   | 0.00     | 5       |
| 3     | 0.00      | 0.00   | 0.00     | 142     |
| 4     | 0.74      | 0.84   | 0.79     | 214     |

_**Table 2. Confusion Matrix**_

|   |   |     |   |     |
|---|---|-----|---|-----|
| 2 | 0 | 4   | 0 | 14  |
| 0 | 0 | 0   | 0 | 5   |
| 1 | 0 | 100 | 1 | 40  |
| 0 | 0 | 0   | 0 | 3   |
| 3 | 0 | 31  | 1 | 179 |

Since we want to analyze the resource usage when the model is trained, we do not pre-process the data and train 
our model with the original data. By doing this, the training will be compute-intensive as well as memory consuming, 
and the impact of model and hyper-parameter choices can be more visible. This may make the bottlenecks to be identified 
and the improved more easily.

When it comes to resource profiling, we focus on CPU time, memory usage and cache used if possible as these usually are 
the possible bottlenecks to be optimized.

### **4. Discussion**

[Add discussion here, cut other sessions off]

## **References**

**[1]**  Roland N Boubela, Klaudius Kalcher, Wolfgang Huf, Christian Naˇsel, andEwald Moser.  Big data approaches for the analysis of large-scale fmri datausing  apache  spark  and  gpu  processing:  a  demonstration  on  resting-statefmri data from the human connectome project.Frontiers  in  neuroscience,9:492, 2016.

**[2]**  Nadine Chang,  John A Pyles,  Austin Marcus,  Abhinav Gupta,  Michael JTarr, and Elissa M Aminoff.  Bold5000, a public fmri dataset while viewing5000 visual images.Scientific data, 6(1):1–18, 2019.

**[3]**  John Darrell Van Horn and Arthur W Toga. Human neuroimaging as a “bigdata” science.Brain imaging and behavior, 8(2):323–331, 2014.

**[4]**  Guangyi  Zhang,  Vandad  Davoodnia,  Alireza  Sepas-Moghaddam,  YaoxueZhang, and Ali Etemad. Classification of hand movements from eeg using adeep attention-based lstm network.IEEE Sensors Journal, 2019.

**[5]**  Xiao Zheng, Wanzhong Chen, Yang You, Yun Jiang, Mingyang Li, and TaoZhang. Ensemble deep learning for automated visual classification using eegsignals.Pattern Recognition, 102:107147, 2020.

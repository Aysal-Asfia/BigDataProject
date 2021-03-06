# Observed objects classification using deep learning and fMRI data

## **Abstract** 

Functional magnetic resonance imaging (fMRI) is one of the most widely used tools to study the neural underpinnings of  
human cognition. Meanwhile, in the field of neuroimaging, increasing spatial and temporal resolutions as well as 
larger sample sizes lead to a rapid increase in the amount of data that needs to be processed in atypical study.  
That is explosion of big imaging data highlights new challenges,such as data sharing, management, and processing, 
as well as reproducibility. 
In this project, we applied deep learning algorithm to classify objects observed by participants 
using fMRI data. In addition, we also investigated potential challenges with big data in classification problem using large fMRI datasets. 
Particularly, we analysed the performance of the algorithm on used dataset in terms of classification metrics as well 
as used resources in data pre-processing, model training, validation and testing.


## **I. Introduction**

Neuroimaging is a relatively new field emerged with the development in neuroscience, psychology and computer science 
which works with brain images to capture and analyze the structures and functions in parts of neural system. 
Nowadays, due to the excellent resolution of fMRI data, the images containing visual objects can be effectively 
recognized using specific patterns derived from this data that are recorded when images are displayed on screen in 
front of the subjects [5].  
With the augmentation of fMRI datasets, increasing spatial and temporal resolutions as well as larger 
sample size lead to a rapid increase in the amount of data that needs to be processed in a typical study [1]. 
For more clarification, by 2015 the amount of acquired neuroimaging data alone, discounting header information 
and before more files are generated during data processing and statistical analysis, exceeded an average of 20GB 
per published research study [3]. However, as the richness of brain datasets continues to grow and the push to place it 
inaccessible repositories mounts, there are many issues to be considered on data processing, data transferring, storage, 
data analysis as well as data sharing [3]. Thus, the authors indicated it is safe to say that human neuroimaging is now, 
officially, a “big data” science [3].
Furthermore, in neuroimaging field in many applications, the problems that neuroscientists as well as 
computer scientists are currently tackling with are not only on algorithms, but also in resource optimization, 
reproducibility and fault-tolerance. 

In this project, apart from model performance, we analysed resource usage in a deep learning application using fMRI data. 
We applied a deep learning algorithm to classify fMRI brain scan images of different participants. 
Subjects involved in the experiments were shown stimuli, which are different images of static objects as well as animated images. 
The fMRI scan data of their brains with different stimuli was recorded in a few time steps. 
Our input data is region of interest voxels (ROI) preprocessed and represented in vector form. 
Preprocessed ROI data of each participant is different in size as well as the number of features, which is the length of the vector. 
Thus, we trained our model with different subject independently with different combinations of time steps and compared the results. 
Our goal was to have an idea of the improvements in the result with different data sizes, combinations of time steps 
as well as the potential bottleneck in deepp learning applications in neuroimaging using large fMRI datasets.


## **II. Materials and methods**. 

### **1. Dataset**

The data we used is from BOLD5000 repository [2], a human functional MRI (fMRI) study. 
In this dataset, 4 participants were scanned and fMRI data were collected while subjects were asked whether they liked, disliked, 
or were neutral about the stimulus images. The images are labeled with ImageNet super categories. 
From these categories, images are labeled as ”Living Animate”, “Living Inanimate”, “Objects”, “Food” or “Geography”. 
An example of label mapping is “Dog” to “Living Animate” and “Vehicle” to “Objects”. 

Features of fMRI images were represented by using ROIs including: ’LHPPA’, ’RHLOC’, ’LHLOC’, ’RHEarlyVis’, ’RHRSC’, ’RHOPA’,’RHPPA’, ’LHEarlyVis’, ’LHRSC’, ’LHOPA’. 
The data of the ROIs was extracted across the time-course of the trial presentation (TR1=0−2s, TR2 =2−4s, TR3=4−6s, TR4=6−8s, TR5=8−10s, TR34=4-8s),
each of the time steps is represented as a vector.
Data sizes of the subjects are 425 MB, 573 MB, 783 MB and 416 MB respectively.
For the first 3 subjects, we have the same number of experiments, which is 1916, while the numbers of features are 1685, 2270, 3104.
For the last subject, the number of experiments is 1122 and the number of feature is 2787.
 

### **2. Algorithm**

For this classification problem with time-series data, we chose Long Short Term Memory (LSTM), which is 
based on recurrent neural network (RNN), as our classifier. 

Our motivation of applying RNNs is extracting higher-dimensional dependencies from sequential data. 
The main characteristic of RNN units is that they have connections not only between the adjacent layers, but also 
among themselves to capture information from previous inputs. 
Even though traditional RNNs can easily learn short-term dependencies; however, they have difficulties 
in learning long-term dynamics due to the vanishing and exploding gradient problems. The main advantage of LSTM is 
addressing the vanishing and exploding gradient problems by learning both long and short-term dependencies [4].

### **3. Method**

The proposed network includes five LSTM layers and one dense layer. In all of the layers, we used _tanh_ 
activation function and set the dropout rate as 0.25. In the last dense layer, we applied _softmax_ activation function to 
predict the probability of each class. We set batch size to 50 and used _Adam_ as the optimizer.

The model was trained with different subjects independently since the number of features of each subject is different.
For each subject, model was trained twice, once with 5 steps from TR1 to TR5 and once with 2 steps combined in TR34.
With 5 time-steps data, as each of these is represented as a vector of features, we stacked all times-steps to create a 2d input array. 
Three fourths of each subject data was used to train, the remaining was test data.
 
During data pre-processing and model training, we collected the information of CPU time, memory usage, disk throughput and cache used. 
Then we compared the results from different subjects as well as results from the same subject with different time-steps combinations.

### **4. Technologies, libraries and tools**

We chose **_tensorflow_**, **_keras_**, which is widely used for deep learning, to implement LSTM algorithm
and **_numpy_**, **_sklearn_** for data pre-processing. 
In order to obtain system information including memory used and disk throughput, we ran **_atop_** linux command 
and **_collectl_**, a daemon software that collects system performance data.
Our implementation was run on a cloud VM on Compute Canada with Centos 7 OS, GenuineIntel 2.6GHz single core processor, 16GB of RAM and a HDD of 220GB.


## **III. Result**

As this is a multi-class classification problem and the classes are imbalance (for example for subject 1, the numbers 
of data of each class are 30, 6, 170, 5, 267), the model performance was first evaluated using multi-class f1 score and accuracy. 
_Figure 1_ describes the scores of the model trained with different subjects and different time steps, from TR1 to TR5 and TR34. 
A glance at the figure shows that for all subjects, the scores of the model trained with TR34 step are higher 
than the those of the model trained with all 5 steps from TR1 to TR5. This may be explained as the important features might 
concentrate in TR34 step, and as we used all time steps from TR1 to TR5, we might be either adding more noise data or making 
our model overfit. 

(Confusion matrices and ROC curves of each subject can be found in [result](/result) folder.)

![](result/score.png) 

_<div align="center">Figure 1. F1 score and accuracy of the model trained with different subject data.</div>_


During the runtime of the pipeline, the system information was collected using _atop_ and _collectl_. The amount of memory used 
collected by _atop_  as well as other information of the subjects are described in _Table 1_. We can see that the bigger input data
our model was trained with, the more memory was used, and the amount of memory used when we trained model with 
data of time-step 3-4 (TR34) was by far smaller than that with all 5 time-steps.


| Subject | Data size | Experiments | Features | Used memory (TR1-5) | Used memory (TR34) |
|---------|-----------|-------------|----------|---------------------|--------------------|
| 1       | 425 MB    | 1916        | 1685     | 1164 MB             | 614 MB             |
| 2       | 573 MB    | 1916        | 2270     | 1383 MB             | 692 MB             |
| 3       | 783 MB    | 1916        | 3104     | 1811 MB             | 818 MB             |
| 4       | 416 MB    | 1122        | 2787     | 1233 MB             | 676 MB             |

_<div align="center">Table 1. Classification measurements report</div>_


We also recorded data processing time as well as training time of subjects as is shown in _Figure 2_. 

![](result/time.png)

_<div align="center">Figure 2. Processing and training time with different input data size.</div>_


As we can see, as input data grows larger, the runtime of the pipeline becomes longer. However, most of the time is training time 
with all subjects and time-steps used. 

To look closer at data processing and training phases, we used data collected by _collectl_ to monitor disk read/write operations.
Figure 3 shows the disk throughput when we ran our pipeline using subject 3 data with 5 time-steps (TR1-5). 
The top graph illustrates the amount of memory used and the bottom graphs shows the disk throughput during the task.  

![](result/csi3/5steps/mem_prof.png)

_<div align="center">Figure 3. Memory used and disk throughput.</div>_


As is shown in the figure, data was only read from disk during data processing phase and there was no disk read/write during training phase. 
This can be explained as when the data size is small enough to fit in memory, there was no need to swap data to disk.

(Memory profiling results of other subjects are available in [result](/result) folder.)


## **IV. Discussion and future work**

In this project, we pre-processed data and trained our model with different subjects and different time-steps. The variation 
in data size as well as features resulted in the variation in performance of the model. Our results showed that 
feature engineering (selection or processing) can not only help to achieve better results, but may optimize resource usage. 

However, we only managed to use preprocessed data, which is regions of interest (ROI) voxels in vector form. In neuroimaging, 
with the development of brain scanning techniques and storage technologies, there have been many deep learning applications and pipelines 
using raw fMRI brain images, which can be up to several GBs in size for each subject, as the input. Using this big data can 
make the data read time considerably long compared to training time, intertwined reading and writing during the pipeline, 
swapping in/out data to/from disk, etc, which we have not seen in _Figure 3_. 

By the time we did this project, there had been a number of big data analytics techniques have been implemented and optimized 
in big data frameworks, but deep learning techniques with big data still remained as a question to us. 
In the future work, we are interested in defining a new deep learning use case using raw fMRI images as the input, which is considerably large 
to investigate the scalability of deep learning techniques with fMRI data.   

## **References**

**[1]**  Roland N Boubela, Klaudius Kalcher, Wolfgang Huf, Christian Naˇsel, andEwald Moser.  Big data approaches for the analysis of large-scale fmri datausing  apache  spark  and  gpu  processing:  a  demonstration  on  resting-statefmri data from the human connectome project.Frontiers  in  neuroscience,9:492, 2016.

**[2]**  Nadine Chang,  John A Pyles,  Austin Marcus,  Abhinav Gupta,  Michael JTarr, and Elissa M Aminoff.  Bold5000, a public fmri dataset while viewing5000 visual images.Scientific data, 6(1):1–18, 2019.

**[3]**  John Darrell Van Horn and Arthur W Toga. Human neuroimaging as a “bigdata” science.Brain imaging and behavior, 8(2):323–331, 2014.

**[4]**  Guangyi  Zhang,  Vandad  Davoodnia,  Alireza  Sepas-Moghaddam,  YaoxueZhang, and Ali Etemad. Classification of hand movements from eeg using adeep attention-based lstm network.IEEE Sensors Journal, 2019.

**[5]**  Xiao Zheng, Wanzhong Chen, Yang You, Yun Jiang, Mingyang Li, and TaoZhang. Ensemble deep learning for automated visual classification using eegsignals.Pattern Recognition, 102:107147, 2020.

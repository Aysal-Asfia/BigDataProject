Observed objects classification based on fMRI data

####**Abstract** 

Neuroimaging is one of the fields that related to big data for some reasons. First of all, with the development of 
MRI scanner, now we can capture images of brain in ultra-high resolution. But this development comes with the issue 
that MRI data become big data that is difficult to be stored and processed by personal computers. In the other hand,
MRI data can be analyzed to structure human brain, research brain functions and activities using different techniques
including machine learning algorithms. In this project, we present a classification problem using fMRI images and 
neural network. We also analyze the performance of the algorithms in terms resource usage.


####**I. Introduction**

Neuroimaging is a relatively new field emerged with the development in neuroscience, psychology and computer science 
which works with brain images to capture and analyze the structures and functions in parts of nervous system.
Thanks for the development of MRI techniques, compute and storage capability of modern computers as well as 
the efficiency machine learning algorithms, nowadays we can capture human brain images in high resolution, 
and use this data to analyze the structure of human brain. When it comes to computer science, using 
machine learning algorithms, now we are able to investigate how brain functions in certain activities. 

In this project, we apply deep learning algorithms to classify objects observed by subjects using fMRI data. 
Subjects involved in the experiments were shown different images of static objects as well as animated images. 
The fMRI data of their brains when they were looking at the images was recorded in time steps.
This is the input for our classification algorithm.

Since the neuroimaging data is considered big data, in many applications, 
the problems that neuroscientists as well as computer scientists are tackling with are not only which algorithms 
to apply, what is the best configuration for the algorithms, but also the problems in resource optimization, 
reproducibility and fault-tolerance.  Thus, in this project, we also profile the performance of the algorithm 
on used data set  in terms of used resources such as CPU and memory in different steps, from data pre-processing 
to model training,  validation and testing. By doing this analysis, our expectation is to have an idea of 
where could be the bottleneck, what could be improved to optimize resource usage.

####**II. Materials and methods**. 
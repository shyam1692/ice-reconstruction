## Neural Networks for Structured Surface Reconstruction
Created by Shyam Narasimhan

Note: This work is a followup of Research work under Professor Crandall
You can find relevant papers in this link: https://arxiv.org/pdf/1801.03986.pdf

# Introduction
We have seen various applications of Deep Learning in classification problems in computer vision.
Here, we use similar principles to train the neural network to reconstruct the Ice bed and air bed boundaries (more about this below)

# Problem Statement
Glaciology has always been an interesting field of study, and the whether and climatic changes can be studied extensively in this field.
One of the applications of Glaciology is to estimate the Ice bed surfaces (or any other kind of bed surfaces for that matter) below the ice sheets of Arctic / Antarctic region.
Long time back, this was done by digging the bed surfaces. But this is inefficient in many terms.

In past few decades, Radar signals are being effectively used to study the underlying structure of the ice sheets.

But the detection of the bed surfaces from Radar output requires human annotation. This is a very tedious job.
Hence, we develop an algorithm to recognize the bed surfaces by supervised learning.


Previously, Markov Random Fields and 3D CNN Networks have been applied for this task.
Now, we explore the application of different forms of C2DN Networks, ranging from simplistic architecture to more complex architecture of using Mask RCNN (work in progress)

# C2DN
The C2D Network has worked as efficiently as C3D network. The average errors in the estimation of Ice bed surface is 23.13 pixels, and for air bed surfaces is 6.16 pixels.

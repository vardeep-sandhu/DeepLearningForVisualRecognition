# DeepLearningForVisualRecognition

This repo contains assignments for the course Deep Learning for Visual Recoginition taught at University of Bonn in WS 2020-21. Below you will find a short summary of the assignment and possibly some visualizations of what we did.

### Sheet 01

In this assignment we implemented, tested and trained k-Nearest Neighbor (kNN) classifier and a linear classifier with the logistic function. We used Fashion-MNIST dataset to check the performance of the classification algorithms.

<p align="center">
<img src="./images/Confusion_mat.png?raw=true" width="40%" height="40%" >

Here you can see the Confusion Matrix of the best model.

### Sheet 04

In this sheet we focus on checking Vanishing Gradients which is a major hinderance while learning. This we do by checking the norms while learning and applying some techniques to overcome this vanishing gradient problem.

Later we also used different optimizers (with adaptive learning rates) and checked their performance on the same Fashion-MNIST dataset.

<p align="center">
<img src="./images/SGD.png?raw=true" width="40%" height="40%" >

### Sheet 05
  
In this sheet the main focus was Normalization and Advanced Regularization. So in the first task we applied Orthogonal matrix regularization which claim that networks benefit from weight matrices which are regularized to be orthogonal.
  
Secondly we performed Label smoothing which is one of the techniques is to apply noise to the output to help the model in acheiving generalization.
  
Lastly, we applied the most widely used Batch Normalization in our model and we later compared the performance resuts of various techiniques.
  
### Sheet 06

By this time so were done with MLPs and finally we got Convolutional Neural Networks. The first task was to make a simple CNN model and train and test it on CIFAR-10 dataset. This model had the bare minimum when it came to architecture and this served as a baseline. 
  
Later, we added data augmentaiton to our training dataset to push the accuracy of our CNN. The reason for this is that the amount of training data we got is limited, so this is a way to get much more images for free
  
Lastly, we visualized the filter response to see how the different filters and layers looked like. Here are some visuals
<p align="center">
<img src="./images/Activation_vis.png?raw=true" width="40%" height="40%" >

### Sheet 07

By this time we were getting a hold of CNNs and the differnt popular archtectures. So this next sheet focused on the architecture which finished the ImageNET challange once and for all, ResNets.

The first task focused on making a small residual network in which we implemented a residual block and then once the ResNet model was done we evaluated it on the CIFAR-10 dataset. Secondly, the final task was to acheive test accuaracy of more than 85% on CIFAR-10 dataset. This was an open problem and we could use any architecture of our choice to make this happen. 

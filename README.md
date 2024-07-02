# Machine_Learning

## Introduction to Machine Learning

A subset of artificial intelligence known as machine learning focuses primarily on the creation of algorithms that enable a computer to independently learn from data and previous experiences. 

Machine learning algorithms create a mathematical model that, without being explicitly programmed, aids in making predictions or decisions with the assistance of sample historical data, or training data. For the purpose of developing predictive models, machine learning brings together statistics and computer science. Algorithms that learn from historical data are either constructed or utilized in machine learning. The performance will rise in proportion to the quantity of information we provide.

A machine can learn if it can gain more data to improve its performance.

## How does Machine Learning work

A machine learning system builds prediction models, learns from previous data, and predicts the output of new data whenever it receives it. The amount of data helps to build a better model that accurately predicts the output, which in turn affects the accuracy of the predicted output.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/0dcc806a-66f3-48e9-993e-1ba3b4e22e7f)


Let's say we have a complex problem in which we need to make predictions. Instead of writing code, we just need to feed the data to generic algorithms, which build the logic based on the data and predict the output. Our perspective on the issue has changed as a result of machine learning. The Machine Learning algorithm's operation is depicted in the following block diagram:

## Features of Machine Learning:

* Machine learning uses data to detect various patterns in a given dataset.

* It can learn from past data and improve automatically.

* It is a data-driven technology.

* Machine learning is much similar to data mining as it also deals with the huge amount of the data.

## Classification of Machine Learning

At a broad level, machine learning can be classified into three types:

1. Supervised Learning

2. Unsupervised Learning

3. Reinforcement Learning

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/c2e5bcbf-4162-4cc7-8196-19b75e33b127)


### 1. Supervised Learning

In supervised learning, sample labeled data are provided to the machine learning system for training, and the system then predicts the output based on the training data.

Supervised learning can be grouped further in two categories of algorithms:

* Classification

* Regression

### Classification

When the output variable is categorical i.e. with two or more classes(yes/no, true/false), we make use of classification.

### Regression

It is used to understand the relationship between dependent and independent variables. When the output variable is real or continuous value, we make use of regression.

Below are some popular algorithms which come under supervised learning:

## Linear Regression

Linear regression is one of the easiest and most popular Machine Learning algorithms. It is a statistical method that is used for predictive analysis. Linear regression makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, etc.

For each type of linear regression, it seeks to plot a line of best fit. However, unlike other regression models, this line is straight when plotted on a graph.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/3ba0f971-7177-467d-b76d-7d004c55209f)


## Logistic Regression

Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/3b4f8598-64ca-4b02-91d8-bfa29425ae65)


## Random Forest

Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.

The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.

The below diagram explains the working of the Random Forest algorithm:

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/c37cbd2e-bdce-4f35-89f4-2889da3d4eec)


## K-Nearest Neighbor

K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/a2ca8ce3-00cc-4715-8896-b90d3d121a1a)


## Support Vector Machine (SVM)

Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.

The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/713308be-d5c3-41d3-b31b-0c3518a1373a)

## Naive Bayes

Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems. It is mainly used in text classification that includes a high-dimensional training dataset.

Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.

## Advantages of Supervised learning:

* With the help of supervised learning, the model can predict the output on the basis of prior experiences.

* In supervised learning, we can have an exact idea about the classes of objects.

* Supervised learning model helps us to solve various real-world problems such as fraud detection, spam filtering, etc.

## Disadvantages of supervised learning:

* Supervised learning models are not suitable for handling the complex tasks.

* Supervised learning cannot predict the correct output if the test data is different from the training dataset.

* Training required lots of computation times.

* In supervised learning, we need enough knowledge about the classes of object.

### 2. Unsupervised Learning

Unsupervised learning is a type of machine learning in which models are trained using unlabeled dataset and are allowed to act on that data without any supervision.

The unsupervised learning algorithm can be further categorized into two types of problems:

## Clustering

Clustering is a method of grouping the objects into clusters such that objects with most similarities remains into a group and has less or no similarities with the objects of another group. Cluster analysis finds the commonalities between the data objects and categorizes them as per the presence and absence of those commonalities.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/7939a605-2b2e-445a-9c0e-7ac88353b726)


Below is the list of some popular unsupervised learning algorithms:

## K-means Clustering

K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. Here K defines the number of pre-defined clusters that need to be created in the process, as if K=2, there will be two clusters, and for K=3, there will be three clusters, and so on.

It allows us to cluster the data into different groups and a convenient way to discover the categories of groups in the unlabeled dataset on its own without the need for any training.

It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/d3531864-5c82-4f1e-aac4-052664781324)


## Hierarchical Clustering

Hierarchical clustering is another unsupervised machine learning algorithm, which is used to group the unlabeled datasets into a cluster and also known as hierarchical cluster analysis or HCA.

In this algorithm, we develop the hierarchy of clusters in the form of a tree, and this tree-shaped structure is known as the dendrogram.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/6d4f4a5a-3485-4f9b-882d-4c870ff6b357)


## Principal Component Analysis (PCA)

Principal Component Analysis is an unsupervised learning algorithm that is used for the dimensionality reduction in machine learning. It is a statistical process that converts the observations of correlated features into a set of linearly uncorrelated features with the help of orthogonal transformation. These new transformed features are called the Principal Components. It is one of the popular tools that is used for exploratory data analysis and predictive modeling. It is a technique to draw strong patterns from the given dataset by reducing the variances.

## Density-Based Clustering

Density-Based Clustering refers to one of the most popular unsupervised learning methodologies used in model building and machine learning algorithms. The data points in the region separated by two clusters of low point density are considered as noise. The surroundings with a radius ε of a given object are known as the ε neighborhood of the object. If the ε neighborhood of the object comprises at least a minimum number, MinPts of objects, then it is called a core object.

![image](https://github.com/prachipaunikar/Machine_Learning/assets/147481200/8120cb43-67fb-458c-8518-d016a3d3b81e)


## Advantages of Unsupervised Learning

* Unsupervised learning is used for more complex tasks as compared to supervised learning because, in unsupervised learning, we don't have labeled input data.

* Unsupervised learning is preferable as it is easy to get unlabeled data in comparison to labeled data.

## Disadvantages of Unsupervised Learning

* Unsupervised learning is intrinsically more difficult than supervised learning as it does not have corresponding output.

* The result of the unsupervised learning algorithm might be less accurate as input data is not labeled, and algorithms do not know the exact output in advance.

### 3. Reinforcement Learning

Reinforcement Learning is a feedback-based Machine learning technique in which an agent learns to behave in an environment by performing the actions and seeing the results of actions. For each good action, the agent gets positive feedback, and for each bad action, the agent gets negative feedback or penalty.

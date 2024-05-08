## 目录
- [1. What's Decision Tree ](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#1什么是线性回归)
- [2. Why Decision Tree](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#2-能够解决什么样的问题)
- [3. Math](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#3-一般表达式是什么)
- [4. Pros and Cons](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#4-如何计算)
  
## 1. What's Decision Tree
A decision tree is a popular supervised learning algorithm used for both classification and regression tasks in machine learning. It works by creating a tree-like structure where each internal node represents a decision based on a feature, and each leaf node represents the outcome or prediction. 

Here's how a decision tree typically works:
Splitting: The algorithm starts at the root node and selects the best feature to split the data based on certain criteria, often aiming to maximize information gain or minimize impurity. This process continues recursively for each child node until a stopping criterion is met, such as reaching a maximum depth or having a minimum number of samples in a node.
Decision Making: At each node, a decision is made based on the feature value. For example, if the feature is categorical (e.g., color is "red" or "blue"), the decision might be based on which category the data point belongs to. If the feature is numerical (e.g., age), the decision might be based on whether the value is above or below a certain threshold.
Leaf Nodes: When a stopping criterion is met (e.g., maximum depth reached or minimum samples in a node), the node becomes a leaf node and represents a class label (in classification) or a predicted value (in regression).
Prediction: To make predictions, new data points are passed down the decision tree, following the decisions at each node until they reach a leaf node, which gives the final prediction.

What's the difference between logistic regression and decision tree: 
Logistic Regression and trees differ in the way that they generate decision boundaries i.e. the lines that are drawn to separate different classes. To illustrate this difference, let’s look at the results of the two model types on the following 2-class problem:

<p align="center" width="100%">
    <img width="40%" src="https://blog.bigml.com/wp-content/uploads/2016/06/model_boundary_radial.png?w=2100&h=&crop=1">
    <img width="40%" src="https://blog.bigml.com/wp-content/uploads/2016/06/lr_boundary_radial.png?w=2100&h=&crop=1">
</p>
<p align="center" width="100%">
    <img width="40%" src="https://almablog-media.s3.ap-south-1.amazonaws.com/image_17_ef951c96c2.png">
    <img width="40%" src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Aowj0Cbk5WvbpRE2FPlX0A.png">
</p>



## 2. Why Decision Tree
- reasons/ what problem it solves
- application

## 3. Math 
- How to grow a tree
- Model expression
- Loss function
- Hyperparameters

## 4. Pros and Cons 
- Bias and variance
- linear and nonlinear
- num of feature

## 5. Reference
- Decision tree Sklearn: https://scikit-learn.org/stable/modules/tree.html


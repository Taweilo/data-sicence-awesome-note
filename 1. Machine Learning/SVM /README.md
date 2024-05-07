# SVM 
- [1. What's SVM ](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#1什么是线性回归)
- [2. Why SVM](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#2-能够解决什么样的问题)
- [3. Math](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#3-一般表达式是什么)
- [3. Math](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#3-一般表达式是什么)
- [4. Pros and Cons](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#4-如何计算)
  
## 1. What's SVM
A support vector machine (SVM) is a supervised machine learning algorithm that classifies data by finding an optimal line or hyperplane that maximizes the distance between each class in an N-dimensional space.

<img src="https://monkeylearn.com/static/7002b9ebbacb0e878edbf30e8ff5b01c/d8712/plot_hyperplanes_annotated.webp" height="350">

## 2. Why SVM
- What problem it solves: <br>
Support Vector Machines (SVM) are powerful models used to classify data points into distinct categories in binary and multiclass classification problems. They employ the kernel trick to handle nonlinear relationships by transforming the input space into a higher-dimensional space, enabling the identification of a linear decision boundary even in non-linearly separable data using functions like linear, polynomial, radial basis function (RBF), and sigmoid kernels. SVMs maximize the margin between different classes, improving generalization and noise robustness, with the decision boundary defined by support vectors, the closest data points to the hyperplane.

<img src="https://media.licdn.com/dms/image/C5612AQGmpMqnj-rzjg/article-inline_image-shrink_1000_1488/0/1616425192695?e=1720051200&v=beta&t=-a9F-FAjgwYrmEBuCqkmXNwAc7xf0RtuGtm4VB6jY5o" height="350">
<img src="https://media.licdn.com/dms/image/C5612AQFdXYFn7brlSw/article-inline_image-shrink_400_744/0/1616425062511?e=1720051200&v=beta&t=HuCOastixJwYTNRKSjcWFGLTAWTlrm6kxPJhDpyvjOw" height="350">

- Application: <br>
A good use case for SVM would be in scenarios where there's a need to classify data into distinct categories with a clear margin of separation, especially when dealing with nonlinear relationships or high-dimensional feature spaces. E.g. Image Classification, Text Classification, Bioinformatics: SVMs are applied in bioinformatics for tasks like protein classification, gene expression analysis, and disease prediction. Financial Forecasting: They are used in financial applications for credit scoring, stock market prediction, and risk management. Medical Diagnosis: SVMs are utilized in medical diagnostics for disease classification, patient outcome prediction, and medical image analysis.

## 3. Math 
- Model expression
Given a training dataset {(𝑥₁, 𝑦₁), (𝑥₂, 𝑦₂), ..., (𝑥ₙ, 𝑦ₙ)} where 𝑥ᵢ is the feature vector and 𝑦ᵢ is the class label (+1 or -1), the hard margin SVM objective can be expressed as:
  - Hard Margin: <br>
    $min_{w, b} (1/2) ||w||^2$ <br>
    Subject to the constraints: <br>
    $y_i (w^T x_i + b) >= 1$ for $i = 1, 2, ..., n$  <br>
  - Soft Margin: <br>
    $min_{w, b, \xi} (1/2) ||w||^2 + C \sum\limits_{i=1}^{n} \xi_i$ <br>
    Subject to the constraints: <br>
    $y_i (w^T x_i + b) >= 1 - \xi_i$ and $\xi_i >= 0$ for $i = 1, 2, ..., n$ <br><br>
    
    The hinge loss $\xi_i$ represents the penalty associated with the misclassification of a data point $(x_i, y_i)$. Here's how the hinge loss is defined:
    $\xi_i = \max(0, 1 - y_i (w^T x_i + b))$ <br>
    Where:
    - $\xi_i$ is the hinge loss for the \( i \)-th data point.
    - $x_i$ is the feature vector for the \( i \)-th data point.
    - $y_i$ is the true class label for the \( i \)-th data point, with \( y_i = +1 \) for the positive class and \( y_i = -1 \) for the negative class.
    - $w$ is the weight vector.
    - $b$ is the bias term.
<img src="https://media.geeksforgeeks.org/wp-content/uploads/20231109124420/1-(1).png" height="350">
      
- Hyperparameters
  - **C** (Regularization Parameter): Controls the trade-off between maximizing the margin and minimizing the classification error. Higher values of C allow for a smaller margin but fewer misclassifications (potentially leading to overfitting), while lower values encourage a larger margin but may lead to more misclassifications.
  - **Kernel Type and Parameters**: For non-linearly separable data, the choice of kernel (e.g., **linear, polynomial, RBF**) and its associated parameters (e.g., degree for polynomial kernel, gamma for RBF kernel) impact the model's ability to capture complex relationships in the data.
  - **Kernel Coefficient**: In the case of polynomial and RBF kernels, the kernel coefficient (gamma) determines the influence of individual training samples on the decision boundary.
## 4. Deployment
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
```
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
```

## 5. Pros and Cons 
- Bias and variance
- linear and nonlinear
- num of feature

## 6. SVR

## 5. Reference
- Sklearn: 


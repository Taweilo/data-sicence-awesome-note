# SVM 
- [1. What's SVM ](#1-What's-SVM)
- [2. Why SVM](#2-Why-SVM)
- [3. Math](#3-math)
- [4. Deployment](#4-deployment)
- [5. Pros and Cons](#5-pros-and-cons)
- [6. SVR](#6-svr)
  
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
<img src="https://editor.analyticsvidhya.com/uploads/20470svm17.png" height="350">

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

<img src="https://cdn-gonif.nitrocdn.com/THQcPJbuTzJiTSVPDIWAOpVBJvtrgqnR/assets/images/optimized/rev-4123e83/training.atmosera.com/wp-content/uploads/2021/07/c-margins-1.png" height="350">

  - **Kernel Type and Parameters**: For non-linearly separable data, the choice of kernel (e.g., **linear, polynomial, RBF**) and its associated parameters (e.g., degree for polynomial kernel, gamma for RBF kernel) impact the model's ability to capture complex relationships in the data.
<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png" height="350">

  - **Kernel Coefficient**: In the case of polynomial and RBF kernels, the kernel coefficient (gamma) determines the influence of individual training samples on the decision boundary.
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*LWUbFLuATUtNWsKC4AcZWg.png" height="350">

## 4. Deployment
Notebook: https://colab.research.google.com/drive/1sl3OyK5rT2HqDsZ2cUg0P7476pGDAKrB#scrollTo=WIDLhNA89Gd2
SKlearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# Step 1: Generate synthetic dataset and plot pair plot
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_clusters_per_class=1, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
df['class'] = y
sns.pairplot(df, hue='class')
plt.suptitle('Pair Plot with Hue: Class')
plt.show()

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create a pipeline with scaling and SVC, and find best hyperparameters using GridSearchCV
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': [0.1, 0.01, 0.001, 0.0001], 'svc__kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Step 4: Print the best hyperparameters and mean accuracy on train CV data
best_svc = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

cv_mean_accuracy = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
print(f'Mean Accuracy on Train CV Data: {cv_mean_accuracy:.2f}')

# Step 5: Retrain SVC with the best hyperparameters on the entire training data
best_svc.fit(X_train, y_train)

# Step 6: Evaluate the model on test data and print classification report
y_pred = best_svc.predict(X_test)
print('Classification Report on test:')
print(classification_report(y_test, y_pred))

# Step 7: Plot ROC AUC curve
y_pred_proba = best_svc.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC AUC Curve (Area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for Best SVC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```

## 5. Pros and Cons 
- Bias and variance
  - High 𝐶  (Low Bias, High Variance):
  With a high value of C, SVM focuses on classifying the training data correctly and allows for fewer misclassifications. This can lead to a model that is more complex and 
has lower bias but higher variance. In other words, it's more prone to overfitting as it tries to fit the training data very closely.
  - Low 𝐶 (High Bias, Low Variance):
Conversely, a low value of C encourages a larger margin and accepts more margin violations (misclassifications). This results in a simpler model with higher bias but lower variance. Such a model is less likely to overfit but may have higher bias, meaning it might not capture the training data as accurately.


## 6. SVR
Unlike Support Vector Machines (SVMs) used for classification tasks, SVR seeks to find a hyperplane that best fits the data points in a continuous space. This is achieved by mapping the input variables to a high-dimensional feature space and finding the hyperplane that maximizes the margin (distance) between the hyperplane and the closest data points, while also minimizing the prediction error.

## 5. Reference
- https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/4.%20SVM/4.%20SVM.md
- SVC Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- SVR Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
- Hard margine: https://www.youtube.com/watch?v=bM4_AstaBZo
- Soft margin: https://www.youtube.com/watch?v=IjSfa7Q8ngs
- Hindge loss: https://www.youtube.com/watch?v=eKIX8F6RP-g&t=0s
- Kernel trick visualization: https://www.youtube.com/watch?v=wqSTBCguVyU
- Kernel trick with different dataset: https://kevinvecmanis.io/machine%20learning/hyperparameter%20tuning/dataviz/python/svm/2019/05/12/Support-Vector-Machines-Visual-Guide.html
- SVR: https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/#:~:text=Support%20Vector%20Regression%20(SVR)%20is,while%20minimizing%20the%20prediction%20error.


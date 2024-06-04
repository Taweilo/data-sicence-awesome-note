# Data for Data Science

## 1. Simulated data
1. regression
```
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=5, n_features=2, noise=1, random_state=42)
```
2. classification
```
from sklearn.datasets import make_classification
X, y = make_classification(random_state=42)
X.shape, y.shape
>>> (100, 20)
```
3. clustering
```
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state=0)
X.shape, y.shape
>>> (10, 2)
```
4. special data type - make circles
```
from sklearn.datasets import make_circles
X, y = make_circles(random_state=42)
X.shape, y.shape
>>> (100, 2)
```
5. special data type - make moons
```
X,y = make_moons( samples=1000, noise=0.03, random_state=42)
X.shape, y.shape
>>> ((1000, 2), (1000,))
```
## 2. Real dataset from Toy libray 
1. Iris plants dataset
   - classification
   - 150 (50 in each of three classes)
   - 4 numeric attributes
2. Diabetes dataset
   - regression
   - 442 instances
   - First 10 columns are numeric predictive values
   - Column 11 is a quantitative measure of disease progression one year after baseline
3. Optical recognition of handwritten digits dataset
   - 1797 instances
   - 64 attributes
4. Linnerrud dataset
   - classification
   - 20 instances
   - 3 attributes
5. Wine recognition dataset
   - classification
   - 178 instances
   - 13 numeric, predictive attributes and the class
6. Breast cancer wisconsin (diagnostic) dataset
   - classification
   - 569
   - 30 numeric, predictive attributes and the class

## 3. Reference
- Toy datasets: https://scikit-learn.org/stable/datasets/toy_dataset.html
- Make regression: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
- Make classification:https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
- Make blobs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
- Make moon: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
- Make circles: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html


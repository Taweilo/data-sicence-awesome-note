## 目录
- [1.What's Linear Regression](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#1什么是线性回归)
- [2. 能够解决什么样的问题](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#2-能够解决什么样的问题)
- [3. 一般表达式是什么](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#3-一般表达式是什么)
- [4. 如何计算](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#4-如何计算)


## 1. What's Linear Regression

- 线性：两个变量之间的关系**是**一次函数关系的——图象**是直线**，叫做线性。
- 非线性：两个变量之间的关系**不是**一次函数关系的——图象**不是直线**，叫做非线性。
- 回归：人们在测量事物的时候因为客观条件所限，求得的都是测量值，而不是事物真实的值，为了能够得到真实值，无限次的进行测量，最后通过这些测量数据计算**回归到真实值**，这就是回归的由来。

## 2. 能够解决什么样的问题

对大量的观测数据进行处理，从而得到比较符合事物内部规律的数学表达式。也就是说寻找到数据与数据之间的规律所在，从而就可以模拟出结果，也就是对结果进行预测。解决的就是通过已知的数据得到未知的结果。例如：对房价的预测、判断信用评价、电影票房预估等。

## 3. 一般表达式是什么

![](https://latex.codecogs.com/gif.latex?Y=wx+b)

w叫做x的系数，b叫做偏置项。
y = \theta^T \times x
$\hat{y}$ = \theta^T x$
$\theta_0 + \theta_1 x_1$
## 4. 如何计算

### 4.1 Loss Function--MSE

![](https://latex.codecogs.com/gif.latex?J=\frac{1}{2m}\sum^{i=1}_{m}(y^{'}-y)^2)


利用**梯度下降法**找到最小值点，也就是最小误差，最后把 w 和 b 给求出来。






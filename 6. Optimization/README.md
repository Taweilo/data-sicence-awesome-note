# Linear Optimization Problem

## Problem Statement
- **Objective:** Maximize \( Z = 20X + 25Y \)
- **Subject to:**
  1. \( 2X + Y \leq 50 \)
  2. \( X + 2Y \leq 40 \)
  3. \( X, Y \geq 0 \)

## Steps to Solve Using Lagrange Multipliers

1. **Formulate the Lagrangian:**

   \[
   \mathcal{L}(X, Y, \lambda_1, \lambda_2) = 20X + 25Y + \lambda_1(50 - 2X - Y) + \lambda_2(40 - X - 2Y)
   \]

   Here, \(\lambda_1\) and \(\lambda_2\) are the Lagrange multipliers for the constraints.

2. **Find the Partial Derivatives and Set to Zero:**

   \[
   \frac{\partial \mathcal{L}}{\partial X} = 20 - 2\lambda_1 - \lambda_2 = 0 \quad \Rightarrow \quad \lambda_1 = \frac{20 - \lambda_2}{2}
   \]

   \[
   \frac{\partial \mathcal{L}}{\partial Y} = 25 - \lambda_1 - 2\lambda_2 = 0 \quad \Rightarrow \quad \lambda_1 = 25 - 2\lambda_2
   \]

   \[
   \frac{\partial \mathcal{L}}{\partial \lambda_1} = 50 - 2X - Y = 0 \quad \Rightarrow \quad 2X + Y = 50
   \]

   \[
   \frac{\partial \mathcal{L}}{\partial \lambda_2} = 40 - X - 2Y = 0 \quad \Rightarrow \quad X + 2Y = 40
   \]

3. **Solve the System of Equations:**

   From the equations \(\lambda_1 = \frac{20 - \lambda_2}{2}\) and \(\lambda_1 = 25 - 2\lambda_2\), we get:

   \[
   \frac{20 - \lambda_2}{2} = 25 - 2\lambda_2 \quad \Rightarrow \quad 20 - \lambda_2 = 50 - 4\lambda_2 \quad \Rightarrow \quad 3\lambda_2 = 30 \quad \Rightarrow \quad \lambda_2 = 10
   \]

   Substituting \(\lambda_2 = 10\) back into \(\lambda_1 = 25 - 2\lambda_2\):

   \[
   \lambda_1 = 25 - 20 = 5
   \]

   Now we solve the system of linear equations for \(X\) and \(Y\):

   \[
   2X + Y = 50
   \]

   \[
   X + 2Y = 40
   \]

   Solve these equations simultaneously:

   Multiply the second equation by 2:

   \[
   2X + 4Y = 80
   \]

   Subtract the first equation from this:

   \[
   3Y = 30 \quad \Rightarrow \quad Y = 10
   \]

   Substitute \(Y = 10\) back into the first equation:

   \[
   2X + 10 = 50 \quad \Rightarrow \quad 2X = 40 \quad \Rightarrow \quad X = 20
   \]

   So, the optimal solution is \(X = 20\) and \(Y = 10\).

4. **Calculate the Optimal Objective Value:**

   \[
   Z = 20X + 25Y = 20(20) + 25(10) = 400 + 250 = 650
   \]

## Shadow Price Explanation

The shadow price (or dual value) is the amount by which the objective function value would increase if the right-hand side of a constraint is increased by one unit, while keeping everything else constant.

In this problem:
- The shadow price for the first constraint (\(2X + Y \leq 50\)) is \(\lambda_1 = 5\).
- The shadow price for the second constraint (\(X + 2Y \leq 40\)) is \(\lambda_2 = 10\).

This means:
- Increasing the right-hand side of the first constraint by one unit (from 50 to 51) would increase the optimal objective value by 5 units.
- Increasing the right-hand side of the second constraint by one unit (from 40 to 41) would increase the optimal objective value by 10 units.

## Summary

- **Optimal solution:** \(X = 20\), \(Y = 10\)
- **Optimal objective value:** \(Z = 650\)
- **Shadow prices:** \(\lambda_1 = 5\), \(\lambda_2 = 10\)

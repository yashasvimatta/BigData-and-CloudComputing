
## Overview
This project demonstrates the implementation of linear regression using the normal equation method and gradient descent on Apache Spark. It involves matrix operations, vector manipulations, and calculation of coefficients for a linear regression model. Additionally, it covers the computation of gradients and predictions for regression tasks.

## Features
- **Linear Regression using Normal Equation:** Efficient computation of regression coefficients using the normal equation.
- **Gradient Descent Optimization:** Implementation of gradient descent to iteratively find the optimal weights.
- **RMSE Calculation:** Root Mean Squared Error (RMSE) computation to evaluate the model's performance.
- **Scalable Computations:** Leverage Apache Spark for distributed processing of large datasets.

## Technical Details

### Libraries and Frameworks
- **Apache Spark:** For distributed data processing and machine learning.
- **Breeze:** For efficient numerical linear algebra operations.
- **Spark MLlib:** For machine learning functionalities such as LabeledPoint and Vectors.

## Installation and Setup
The following dependencies are used in the project:
- Apache Spark
- Breeze
- Spark MLlib

## Usage
1. **Initialize Spark Context:**
   Ensure Spark context (`sc`) is initialized before running the cells.

2. **Run Notebook Cells:**
   Execute the notebook cells in order to perform the following:
   - Load and preprocess data.
   - Compute regression coefficients using the normal equation.
   - Perform gradient descent to optimize weights.
   - Evaluate model performance using RMSE.

3. **Outputs:**
   The notebook prints intermediate results such as matrices, vectors, coefficients, gradients, and error metrics.

This project involves applying various statistical and machine learning techniques to analyze the MNIST dataset and a regression dataset. The report evaluates multiple models and approaches for classification and regression tasks, exploring concepts like error rate analysis, principal component analysis (PCA), clustering, and predictive modeling using regression.

The results showcase error rates, reconstruction accuracy, and trade-offs in bias-variance to identify the optimal models for classification and regression.

Structure of the Project
1. K-Nearest Neighbors (KNN) Classification
Purpose: Find the optimal value of 
𝑘
k for KNN and calculate classification error rates for the MNIST dataset.
Key Steps:
Use different 
𝑘
k-values (𝑘=1,2,3,10,100, etc.).
Calculate the error rate and plot the relationship between error rates and 
𝑘
k-values.
Evaluate class-specific true positive rates (TPRs) for different 
𝑘
k-values.
2. Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
LDA:
Fit an LDA model to the MNIST dataset.
Evaluate the model with class-specific TPRs and overall error rates.
Analyze the discriminant functions and prior probabilities.
QDA:
Fit a QDA model.
Calculate the error rates and compare class-specific TPRs with LDA.
3. Naive Bayes Classification
Purpose: Train and evaluate a Naive Bayes classifier on the MNIST dataset.
Key Steps:
Train the model using Gaussian Naive Bayes.
Evaluate the error rates and TPRs for each class.
Findings:
Discuss the limitations of Naive Bayes in high-dimensional datasets.
Part 2: Principal Component Analysis (PCA) on Digits 1 and 8
1. PCA Preprocessing and Optimization
Goal: Reduce the dimensionality of the MNIST dataset for digits 1 and 8.
Steps:
Center the data by subtracting means.
Optimize the PCA problem to estimate principal component directions 
Compute cumulative variance explained by principal components.
2. PCA for Data Reconstruction
Objective: Reconstruct the MNIST images using different numbers of principal components.
Steps:
Reconstruct images for digits 1 and 8 using increasing numbers of principal components.
Evaluate reconstruction errors and visualize their decrease with more components.
Part 3: Clustering Analysis
1. Hierarchical Clustering
Goal: Group data points for digits 1 and 8 using hierarchical clustering.
Techniques:
Explore complete, average, and single linkage clustering.
Visualize dendrograms for each method.
Evaluate clustering quality using contingency tables.
2. 
𝑘
k-Means Clustering
Objective: Apply 𝑘
k-means clustering to group data for digits 1 and 8.
Approach:
Perform clustering on raw data and reduced PCA data.
Compare clustering quality across methods.
Part 4: Regression Modeling
1. Variable Selection (Forward and Backward Selection)
Purpose: Identify the best subset of predictors for regression.
Key Steps:
Perform forward and backward selection using BIC as the criterion.
Evaluate test mean squared error (MSE) for models with different numbers of predictors.
2. Ridge and Lasso Regression
Objective: Address multicollinearity and high dimensionality with penalized regression methods.
Steps:
Tune the penalty parameter (
𝜆
λ) using cross-validation.
Evaluate test MSE and compare results between Ridge and Lasso regression.
3. Principal Component Regression (PCR)
Goal: Use principal components for regression to reduce dimensionality.
Steps:
Apply PCR and determine the optimal number of components.
Evaluate the test MSE for models with varying numbers of components.
4. Partial Least Squares (PLS) Regression
Purpose: Use PLS to model relationships between predictors and responses.
Steps:
Optimize the number of components for PLS.
Evaluate test MSE and compare with PCR.
5. Bias-Variance Trade-off Analysis
Compare test MSEs across different regression techniques.
Discuss the impact of penalization, dimensionality reduction, and subset selection.
Conclusion
The best classifier is KNN with k=1 or k=3, achieving an error rate of 0.046.
For regression, Principal Component Regression (PCR) achieves the lowest test MSE, highlighting its effectiveness in high-dimensional settings.
The project provides insights into model selection, bias-variance trade-offs, and the importance of dimensionality reduction in machine learning.
Usage Instructions
Dataset: Ensure the MNIST dataset and regression datasets are placed in the specified file paths.
Dependencies:
R packages: FNN, e1071, caret, MASS, leaps, glmnet, pls.
Execution:
Each section of the code is self-contained. Run the code blocks sequentially for full reproducibility.
Results:
Review error rates, clustering results, and regression outcomes as visualized in the plots and printed summaries.

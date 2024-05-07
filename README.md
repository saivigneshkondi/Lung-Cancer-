Title of the project: Predictive Analysis of Lung Cancer Using Stochastic Gradient Boosting Along with Recursive Feature Elimination

Steps involved in the project:

Data Collection and Preprocessing:
Gather a dataset containing relevant features (such as patient demographics, medical history, and imaging data) along with the target variable (lung cancer diagnosis).
Clean the data by handling missing values, outliers, and ensuring consistency.
Split the dataset into training and validation sets.

Feature Engineering:
Extract meaningful features from raw data. For lung cancer prediction, features might include:
Radiological features from CT scans (nodule size, shape, texture, etc.).
Clinical features (age, gender, smoking history, etc.).
Normalize or standardize features to ensure consistent scaling.

Model Selection:
Choose stochastic gradient boosting (SGB) as the predictive model.
We have Selected Model SVM for optimised results. 
SGB combines gradient boosting with stochastic gradient descent for better performance.
Other models (such as random forests, logistic regression,Svm) can be considered for comparison.


Model Training:
Train the SGB model and SVM model on the training data.
Optimize hyperparameters (learning rate, tree depth, etc.) using cross-validation.
Monitor performance metrics (e.g., AUC-ROC, accuracy) during training.

Model Evaluation:
Evaluate the SGB model on the validation set.
Assess performance using appropriate metrics (precision, recall, F1-score).
Visualize feature importances to understand which features contribute most to predictions.

Model Interpretation:
Interpret the SGB model by analyzing feature importance scores.
Understand how specific features impact the prediction of lung cancer.

Hyperparameter Tuning:
Fine-tune hyperparameters further to improve model performance.
Use techniques like grid search CV.
Hyperparameters are the variables that the user specify usually while building the Machine Learning model. thus, hyperparameters are specified before specifying the parameters or we can say that hyperparameters are used to evaluate optimal parameters of the model. 
Hyperparameters for a model can be chosen using several techniques such as Random Search, Grid Search, Manual Search, Bayesian Optimizations, etc. In this article, we will learn about GridSearchCV which uses the Grid Search technique for finding the optimal hyperparameters to increase the model performance.

Model Deployment:
Once satisfied with the model’s performance, deploy it in a clinical setting.
Monitor its performance over time and update as needed.

Overview and result of project:
Visualizing AGE column,Visualizing Categorical Columns,Visualizing AGE vs Categorical Columns,heatmap,training and testing,model building,svc(99%),randomforest(95%),gbc(94%),Knn(93%),Logistic regression(84%).To get reduce the dimensionality of the dataset the PCA can be used,28 model analysis in the machine learning,graph between Accuracy vs model ,Time taken vs model,grid search cv on the DecisionTreeClassifier,lgbm(95%).

Comparsion of the all models best accurate result comes from svm (99%)
final selected model is Support Vector Machine(svm) 

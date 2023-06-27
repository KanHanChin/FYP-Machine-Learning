# FYP-Machine-Learning

# Introduction
Coronary heart disease (CHD) is a disease that means when heart's blood supply is blocked or interrupted by a build-up of fatty substances in the coronary arteries. Over time, the walls of arteries can become furred up with fatty deposits. This disease is usually caused by improper lifestyles such as smoking and drinking excessive amounts of alcohol. People who suffer from high cholesterol, high blood pressure, or diabetes will be more easily infected by CHD. CHD is one of the most dangerous diseases in the world. It is because CHD will cause death, it even may lead to many side effects after the patient was saved from this disease. Examples of the side effect are chest pain, weakness, light-headedness, nausea, cold sweat, pain or discomfort in the arms or shoulder, and shortness of breath. It is unfortunate to say that this kind of horrible disease is uncurable. When people are suffered from this disease, they can just manage the symptoms by treatments such as regular exercise, healthy eating, medicines, and surgery. To invent a better diagnostic system to identify CHD, machine learning algorithms will help to solve this problem. Logistic Regression, Decision Tree, and XG Boost are used after considering.

# Methodology & Result
## Data Collection 
The dataset chosen to use in this project is a dataset imported from the Kaggle website. The source is from a cardiovascular study on the residents in Framingham, Massachusetts. It consists of 15 independent variables and a dependent variable. 4273 rows of cases are included in the dataset. The dependent variable named “TenYearCHD” which represents whether a patient will be infected with CHD in ten years. It is a binary variable that only can be 2 outcomes which are infected or not infected with CHD. Independent variable included 6 binary variables, 2 integer variables, 6 continuous variables and 1 ordinal variables.
## Data Analysing & Pre-processing
Data analysing is a process that helps to have a better understanding of this dataset. For examples, the central tendency table, EDA, and correlation coefficient are used in the project. Meanwhile, Data Pre-processing is a process to convert the raw data into a more reasonable and useful data. Data cleaning is one of the ways to pre-process the data. Data cleaning like handling missing values and removing outliers are used in the project. The missing values was replacing by the mean values of each of variables and outliers were removed row by row. The feature boxplot in SPSS helped to locate the outliers. As the result, the raw data with 4232 rows converted into 3275 rows after data cleaning.
## Data Splitting 
Data Splitting is a step to separate the data into two which named as train data and test data. The train data is used to build up a machine-learning model. The test data is used to evaluate algorithms. In the report, the data was split by using Python with the ratio of 8:2. The train data with 80% contains 2620 rows and the test data with 20% contains 656 rows.
## Data Modelling
1.	Logistic Regression
Logistic Regression also known as the logit model is a statistical model that is usually used in case classification or prediction. There are 2 types of Logistic Regression which are Binary Logistic Regression and Multinomial Logistic Regression. In this project, Binary Logistic Regression is used. the independent variable only can be 0 or 1. For example, the patient has or has not practiced cigarettes. Logistic Regression can use to calculate the impact of changes, predict the future values, and classify the weight of independent variables.  
2.	Decision Tree 
Decision Tree is a type of machine learning algorithms, which utilized for classification and regression problems. In this project, classification decision tree is used. Decision Tree can handle both category and numerical data and are simple to interpret. The basic idea of decision tree is to recursively split the data into subsets according to the values of various attributes, while minimizing the impurity of the resulting subsets. The algorithm selects the attribute that best separates the data depending on information gain or Gini at each step of the tree.
3.	XG Boost
XG Boost is a type of machine learning which combines multiple weak prediction models like decision tree and linear models into an advanced model. The invention of XG Boost is to improve the accuracy of a model by focusing on the errors made by the models before combinations. XG Boost are expected to be more efficient and scalable. It can handle a large dataset and used for both regression and classification problems. XG Boost also can handle missing values with the technique known as “sparsity-aware split finding”. Besides, XG Boost can utilize the multiple CPU cores to parallelize the process of train and test. Cross-validation also included in XG Boost to assist in fine-tuning the model hyperparameters.
## Evaluation of Model
1.	Confusion Matrix 
Confusion Matrix	Predicted Value
	Negative	Positive
Actual Value	Negative	True Negative	False Positive
	Positive	False Negative	True Positive
Confusion Matrix is a matrix used to evaluate the performance of a classification model. It shows the number of correct and incorrect predictions made by the model compared to the actual outcomes. True Positive is the predicted positive data is true by comparing with actual value and False Positive is false predicted positive data. True Negative is the predicted negative data is true by comparing with actual value and False Negative is false predicted negative data.
By using the value computed on confusion matrix, the metrics below can be calculated. Recall, Precision, Accuracy, and F1-Score used to evaluate the performance of classification model.
a.	Recall
Recall also known as sensitivity. The proportion of actual positive cases correctly identified by the model is measured by recall. It computes the percentage of true positives (TP) among all actual positives (TP + FN). A high recall value indicates that the model is effective at detecting positive cases.
b.	Precision
Precision is the proportion of positive cases correctly identified by the model is measured by precision. It computes the percentage of true positives (TP) among all predicted positives (TP + FP). A high precision value indicates that the model is effective at detecting only the relevant positive cases.
c.	Accuracy
Accuracy calculates the proportion of all correct predictions (TN+TP) among predictions made (TN+FN+TP+FP). A high accuracy value indicates that the model is good at making correct predictions overall.
d.	F1-Score
The F1-score is a popular metric in classification tasks for assessing a classifier's overall performance. It is the harmonic mean of precision and recall. F1-score has a range from 0 to 1, where 1 is the best possible value and 0 is the worst. A high F1-Score indicates perfect precision and recall and deficient performance of the classifier when F1-Score is low.
The conclusion below will show the metrics of each machine learning algorithms.
## Feature Selection
In real life, it is almost impossible to see that all the variables in a dataset are useful in data modelling. However, some of the variables will decrease a model’s accuracy and capability. So, feature selection is an important step in every models construction. In the report, each of the machine learning algorithms will fit in and obtain the feature scores or p-value to visualize which variables are more important. Feature selection for logistic regression model done by using p-value. The lower the p-value, the more important the variable is. Meanwhile, decision tree and XG Boost cannot obtain p-value, so feature score is used. The variables is much significance when the feature score is higher. 

# Conclusion 
## Key Factor 
Model	Top 1	Top 2	Top 3	Top 4	Top 5
Logistic Regression	Male	Age	sysBP	cigsPerDay_1	prevalentStroke
Decision Tree	Age	sysBP	totChol_1	BMI_1	heartrate_1
XG Boost	Age	Male	BPMeds_1	prevalentHyp	sysBP
From the table above, the top 3 key factors are age, male, and sysBP. By observing the above 3 tables, the features “sysBP”, and “age” always in the top 5 lists, feature “male” happened in the top 5 lists twice. In the conclusion, features “sysBP”, “age”, and “male” are the key factors that will affect the probability of getting CHD in 10 years. 
## Finalize Model
Model	Accuracy	Recall	Precision	F1-Score
Logistic Regression	70%	65%	27%	38%
Decision Tree	79%	36%	30%	33%
XG Boost	91%	99%	62%	76%
The 3 types of machine learning algorithms which are Logistic Regression, Decision Tree and XG Boost have been applied to the dataset. At first, every model has a dissatisfactory result. The confusion matrix cannot get a high value of true positive result. It means that the ability to predict a person whether he/she will suffer from CHD in 10 years is extremely low. After the adjustment of threshold score, the models have becoming more reasonable. Model Logistic Regression has an accuracy 70% and XG Boost model has an accuracy of 91%. Meanwhile Decision Tree model has accuracy of 79% but the recall value is lower than Logistic Regression Model. In the conclusion, the most suitable model is XG Boost model which have the highest accuracy, recall, precision and f1-score. For the future works of this project, Random Forest algorithm will be tried to apply into this dataset to gain the more accurate and suitable model by comparing with Logistic Regression Model.

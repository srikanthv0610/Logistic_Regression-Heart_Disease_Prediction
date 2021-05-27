# Heart Disease Prediction using Logistic Regression:
Usin the heart dataset [here](https://github.com/srikanthv0610/Logistic_Regression-Heart_Disease_Prediction/blob/main/Dataset), I have designed this model for:
* Analysing the different variables and finding relations and insights using descriptive statistical analysis technique
* Data Preprocessing
* Splitting the data into train and test sets
* Using Logistic Regression algorithm to train a model and to predict heart disease
* Analysing the Models performance

## Introduction:
Factors that influence heart disease are body cholesterol levels, smoking habit and obesity, family history of illnesses, blood pressure, and work environment. Machine learning algorithms play an essential and precise role in the prediction of heart disease. 

Heart disease can be predicted based on various symptoms such as age, gender, heart rate, etc. Consequently, this would help to reduce the death rate of heart patients.

## Observing the features with respect to the target variable:
![Calegorical_values](https://github.com/srikanthv0610/Logistic_Regression-Heart_Disease_Prediction/blob/main/Plots/Figure_2.png)

Observation from the above plot:
* cp (Chest pain): People with cp 1, 2, 3 are more likely to have heart disease than people with cp 0.
* restecg (resting EKG results): People with a value of 1 (reporting an abnormal heart rhythm) are more likely to have heart disease.
* exang (exercise-induced angina): people with a value of 0 have more heart disease than people with a value of 1.
* slope (the slope of the ST segment of peak exercise): People with a slope value of 2 (Downslopins: signs of an unhealthy heart) are more likely to have heart disease than people with a slope value of 0 (Upsloping: best heart rate with exercise) or 1 (Flatsloping: minimal change).
* ca (number of major vessels stained by fluoroscopy): the more blood movement the better, so people with ca equal to 0 are more likely to have heart disease.
* thal {thalium stress result}: People with a thal value of 2 (defect corrected: once was a defect but ok now) are more likely to have heart disease.

![Continuous_values](https://github.com/srikanthv0610/Logistic_Regression-Heart_Disease_Prediction/blob/main/Plots/Figure_3.png)

Observation from the above plot:
* trestbps: resting blood pressure anything above 130-140 is generally of concern
* chol: greater than 200 is of concern.
* thalach: People with a maximum of over 140 are more likely to have heart disease.
* the old peak of exercise-induced ST depression vs. rest looks at heart stress during exercise an unhealthy heart will stress more.


## Correlation Matrix and Correlation with target:
![Heatmap](https://github.com/srikanthv0610/Logistic_Regression-Heart_Disease_Prediction/blob/main/Plots/Figure_5.png)

![Heatmap](https://github.com/srikanthv0610/Logistic_Regression-Heart_Disease_Prediction/blob/main/Plots/Figure_6.png)

Observations from correlation:
* fbs and chol are the least correlated with the target variable.

## Results:
![Model Performance](https://github.com/srikanthv0610/Logistic_Regression-Heart_Disease_Prediction/blob/main/Plots/Performance%20Result.PNG)

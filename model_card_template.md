# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained using scikit-learn with 100 estimators 
and a random state of 42. It was developed to predict whether an individual earns more 
or less than $50,000 per year based on Census Bureau data.

## Intended Use
This model is intended for educational purposes to demonstrate how to build and deploy 
a machine learning pipeline using FastAPI. It predicts income level (>50K or <=50K) 
based on demographic and employment features.

## Training Data
The model was trained on the UCI Census Bureau dataset (census.csv), which contains 
demographic information such as age, workclass, education, marital status, occupation, 
race, sex, and native country. 80% of the data was used for training.


## Evaluation Data
The remaining 20% of the Census Bureau dataset was held out as a test set for evaluation. 
The same preprocessing pipeline was applied to the test data using the encoder fitted 
on the training data.

## Metrics
The model was evaluated using precision, recall, and F1 score.
- Precision: 0.7419
- Recall: 0.6384
- F1: 0.6863

## Ethical Considerations
The dataset contains sensitive demographic features such as race, sex, and native country. 
Care should be taken when using this model in any real-world application, as it may 
reflect historical biases present in the Census data. This model should not be used to 
make decisions that affect individuals without careful review of fairness and bias.

## Caveats and Recommendations
This model was built for educational purposes and has not been optimized for production 
use. Hyperparameter tuning, additional feature engineering, and bias auditing are 
recommended before any real-world deployment. Performance varies across demographic 
slices as shown in slice_output.txt.
!pip install yellowbrick

from yellowbrick.model_selection import LearningCurve
from sklearn.model_selection import ShuffleSlipt

# Creating cross validation
cv = ShuffleSplit(n_split = 10, test_size = 0.2, random_state = 0) 

# Creating Learning curve visualizer
visualizer = LearningCurve(model, cv = cv) #model already defined

visualizer.fit(X,y)
visualizer.show()

#Types of Learning Curve
#Bad Learning Curve due to High Bias
#1) Problem :
#The training and testing curves converge at a lower score i.e. higher error.
#No matter how much data is feed into the model, the model will not represent the underlying relationship and has a high error.
#Poor fit with Training Data set.
#Poor generalization as a model has not captured enough information.
#This model can be termed as “Underfitted Model.”
#2) Solution :
#Train longer by getting more training data so that cross validation curve gets more data to converge. A balance needs to occur between spending time by running a low complexity model on more data and increasing the complexity of the model.
#Train with more complex model (e.g. kernelize, non-linear model, ensembled models).
#Get more features into the dataset. Explore the possibility of ‘Omitted Variable Bias’.
#Decrease regularization.
#Try using different cross-validation strategies, revisit preprocessing, study class imbalance, etc.
#New model architecture (Boosting)
#Bad Learning Curve due to High Variance
#1) Problem:
#When there is a large gap between the final converging point of the curve.
#The model is not predicting consistently on the new data set.
#This model can be termed as Overfitted if bias is very low and the gap is high.
#2) Solution:
#Get more data.
#Decrease the number of features.
#Increase regularization.
#New model architecture (Oversampling, Bagging)
#Reduce model complexity (complex models are prone to high variance)
#Ideal Learning Curve
#The model generalizes to new data.
#Train and Test learning curves converge at similar values.
#Smaller the gap between the learning curves, better is the model.

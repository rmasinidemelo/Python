Hands-on implementation
Let's learn more by implementing a dataset. To begin with, I will start by importing the required libraries.

#Importing the Required Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus as pdot
from IPython.display import Image

#Load the data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#Separate the input features and target
X = cancer.data
y = cancer.target
#To begin with, I will start by creating a Decision Tree without any hyperparameters specified. 
#That is, I will use the default values of the parameters. When I say default, some important parameter values to remember are:
#Impurity Measure (criterion) : gini (CART algorithm)
#max_depth = None (maximum level up to which Decision Tree can be grown)
#min_sample_split = 2 

from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(random_state=42)

#Perform the Fit on Training Data
clf_tree.fit(X_train,y_train)
      
#Create ODT file

export_graphviz(clf_tree,
               out_file = "clf_tree.odt",
               feature_names = cancer.feature_names,
                class_names = cancer.target_names,
               filled = True)

#Generate the Graph
clf_tree_graph = pdot.graphviz.graph_from_dot_file("clf_tree.odt")

#Writing to a jpeg file
clf_tree_graph.write_jpg("clf_tree.jpeg")

#Displaying the Image
Image(filename = "clf_tree.jpeg")

clf_tree.get_depth()

#Without the specification of any hyperparameters, the depth of the Decision Tree is 7. Let's evaluate the tree's performance on Training and Validation Data. For our analysis, I am using the roc_auc_score.

from sklearn.metrics import roc_auc_score
y_train_pred = clf_tree.predict(X_train)
roc_auc_score(y_train,y_train_pred)

#Oh!! Great. I have got the roc_auc_score on my training data as 1. That means, as per this value, my model can predict whether breast cancer images are Malignant or Benign with 100% Accuracy. But, there is a high chance that this model has overfitted the data. A better way to evaluate the model is with the use of the Cross-Validation technique. So, let's run the cross-validation.

from sklearn.model_selection import cross_val_score
cross_val_score(clf_tree,X_train,y_train, cv = 5, scoring = "roc_auc")

#As we can see, on average, our model's score is around 93%. Now let's see if we can further improve the model's performance by adjusting the Hyperparameters. To do that, we make use of the sklearn's GridSearchCV module.
      
from sklearn.model_selection import GridSearchCV

parameters = [{"max_depth" : range(2,8), 
               "min_samples_leaf": range(3,10),
               "max_leaf_nodes":range(5,15)}]

clf_tree = DecisionTreeClassifier(random_state = 42)

clf = GridSearchCV(clf_tree,
                  parameters,
                  cv = 5,
                  scoring = "roc_auc")

clf.fit(X_train,y_train)

#So, this has been completed. Let's check the best parameters of the model.
clf.best_params_

#Now, with this model parameters, let's initialize a model and perform the evaluation on Train and Test data.

      
best_tree = DecisionTreeClassifier(random_state = 42,
                                   max_depth = 3, 
                                   max_leaf_nodes=7,
                                   min_samples_leaf=8)
best_tree.fit(X_train,y_train)

cross_val_score(clf,X_train,y_train,scoring="roc_auc",cv = 5)

#So, we now have the roc_auc score as 0.96 on training data. We can verify this by computing the average value of previous scores.

np.mean(cross_val_score(best_tree,X_train,y_train,scoring="roc_auc",cv = 5))

#Now, let's evaluate on our test data.

y_pred_test = best_tree.predict(X_test)
roc_auc_score(y_test,y_pred_test)

#Cool... We have reached a roc_auc score of 0.9361 on test data. That means, my model can predict whether breast cancer images are Malignant or Benign with 93.61% accuracy. Now, before ending, let's visualize the Decision Tree for the best model.

      
#Create ODT file

export_graphviz(best_tree,
               out_file = "best_tree.odt",
               feature_names = cancer.feature_names,
                class_names = cancer.target_names,
               filled = True)

#Generate the Graph
clf_tree_graph = pdot.graphviz.graph_from_dot_file("best_tree.odt")

#Writing to a jpeg file
clf_tree_graph.write_jpg("best_tree.jpeg")

#Displaying the Image
Image(filename = "best_tree.jpeg")

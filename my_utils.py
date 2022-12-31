import numpy as np
import pandas as pd
#import seaborn as sns
import graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.tree import export_graphviz
from sklearn.model_selection import validation_curve

def deal_with_na(df, fill_val=0.0, drop=True):
    """ 
    Function to deal with NaN values in a dataframe. 
    
    :df: DataFrame to be examined
    :fill_val: 0 by default
    :drop: True by default
    
    :returns: Replaces or drops NaN inplace
    """
    
    if df.isnull().any(axis=None):
        if drop:
            print("Number of NaN values: ", df.isnull.sum().sum())
            df.dropna(inplace=True)
        else:
            df.fillna(fill_val, inplace=True)
            
def plot_validation_curve(train_scores, test_scores, param_range, model_name, xlabel):
    """
    Function to plot validation curve 
    
    :train_scores: Training scores obtained after using validation_curve function
    :test_scores: Test scores obtained after using validation_curve function
    :param_range: Parameter range passed to validation_curve function
    :model_name: Model name to be shown at the title of the plot
    :xlabel: Label of the x-axis
    
    :returns: Nothing, plots the validation curve according to the given parameters
    """
    
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)

    plt_title = "Validation curve of " + model_name

    plt.figure()
    plt.title(plt_title)
    plt.xlabel(xlabel)
    plt.ylabel("Score")
    plt.xlim(np.min(param_range)-1, np.max(param_range)+1)

    plt.plot(param_range, train_score_mean, label="Training score", color="darkorange", lw=2)

    plt.fill_between(
        param_range,
        train_score_mean - train_score_std,
        train_score_mean + train_score_std,
        alpha=0.2,
        color="darkorange",
        lw = 2)

    plt.plot(param_range, test_score_mean, label="Cross-validation score", color="navy", lw=2)

    plt.fill_between(
        param_range,
        test_score_mean - test_score_std,
        test_score_mean + test_score_std,
        alpha=0.2,
        color="navy",
        lw = 2)

    plt.legend(loc="best")

    plt.show()

def plot_decision_tree(clf, features, classes):
    """
    Plots the decision tree using the dot file created
    
    :clf: Classifier object, usually a DecisionTree
    :features: List of names of the features
    :classes: List of names of the classes
    
    :returns: graphviz.Source object
    """

    export_graphviz(clf, out_file='temp.dot', feature_names=features, class_names=classes, filled=True, impurity=False)

    with open("temp.dot") as f:
        dot_graph = f.read()

    return graphviz.Source(dot_graph)

def plot_feature_importance(clf, feature_names):
    """
    Plots the feature importance in a horizontal bar plot
    
    :clf: Classifier object
    :feature_names: List of the feature names
    
    :returns: Nothing, plots the horizontal bar graph
    """
    
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)

    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.title("Feature importance plot")
    plt.yticks(np.arange(c_features), feature_names)
    

def corr_cols(corr_matrix, threshold):
    """
    Calculates the correlated columns in a DataFrame
    
    :corr_matrix: The DataFrame generated after running DataFrame.corr()
    :threshold: Threshold of the correlation
    
    :returns: A list of column names with correlation above the given threshold
    """
    
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])
    
    return drop_cols

def data_scaling(est, X, y, scaler):
    """
    Helper function in scaling example
    
    :est: Estimator class
    :X: Feature dataframe
    :y: Target dataframe
    :scaler: Feature scaling algorithm
    
    :returns: Dictionary with model name, scaler name, train scores, and test scores
    """
    #Split the data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    # Make pipeline with the scaler and estimator
    pipe = make_pipeline(scaler, est)
    # Scale the training data
    pipe.fit(X_train, y_train)
    #Gather the scores in a dict
    score_dict = {'model' : est.__class__.__name__,
               'Scaler': scaler.__class__.__name__,
               'Train_score' : pipe.score(X_train, y_train),
               'Test_score' : pipe.score(X_test, y_test)
                }
    
    return score_dict



def plot_learning_curve(est, X_train, y_train, train_sizes, ax, model_name):
    """
    Function to plot training curve 
    
    :est: Estimator object
    :X_train: Training feature dataset obtained from train_test_split
    :y_train: Training class dataset obtained from train_test-split
    :train_sizes: List of training sizes to go into learning_curve
    :ax: Matplotlib axis object
    :model_name: Model name to be shown at the title of the plot
    
    :returns: Nothing, plots the learning curve according to the given parameters
    """
    
    #Calculate the train sizes, scores and test scores using the learning_curve function
    train_sizes, train_scores, test_scores = learning_curve(est, X_train, y_train,
                                             cv=10, train_sizes=train_sizes)
    
    # Calculate training and test mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the learning curve
    ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training')
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation')
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    
    ax.set_title('Learning Curve for {}'.format(model_name), fontsize=15)
    ax.legend(loc='lower right', fontsize=15)


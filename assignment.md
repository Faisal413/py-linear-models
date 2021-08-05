## Python Linear Models Intro

## Introduction

In this assignment, we will work through a linear regression problem from start to finish.

You are given a dataset that contains information about 400 individuals' credit card and bank balances.
Your task is to predict an individual's balance based on various variables.

Note: Web searches are very helpful. Since the main library we are using is "SciKit Learn",
      try including that in your searches. SciKit Learn's documentation is very
      detailed and complete. Here are two links that will be helpful:

          SciKit Learn User Guide: https://scikit-learn.org/stable/user_guide.html
          SciKit Learn API: https://scikit-learn.org/stable/modules/classes.html

## Basic

### Part 1: Prepare data

1. Load the data into a dataframe from `data/balance.csv`.

1. Since `Gender`, `Married` and `Student` are boolean variables, we will skip using these for now.
   Make a new dataframe with these features removed.

1. Since `Ethnicity` is a categorical variable, we'll omit it for now, as well. Make a new dataframe,
   omitting this feature, too.

1. Experiment with the various methods available to you in order to get a feel for the data.
   Hint: try `head()`, `describe()`, and others.

1. Make a scatter matrix (sometimes called
   a "pair plot") of the variables. If you recall, `scatter_matrix` is provided by Pandas.
   Comment on the distribution of your variables,
   and describe the relationships between your numeric feature variables and `Balance`.
   Pay attention to features that may be colinear ("multi-colinearity" - dependence of a feature
   on another feature or features - can lead to poor results in linear models).

1. Optionally, make a "heat map" to further investigate the correlations between
   features, `Balance`, and other features. This code will do this:

   ```python
   sns.heatmap(df.corr(), annot=True, fmt='0.2f', cmap='Purples');
   ```

1. Separate the features from the target, using `X` for the features and `y` for the target (`Balance`).

1. Do `X` and `y` have the same dimensions? You can use Numpy's `shape` property here. If the
   dimensions are different, why is this?

### Part 2: Fit initial model

1. Create a new `LinearRegression` model. This can be done like this:
   ```python
   model = LinearRegression()
   ```
   Note that we are not passing any arguments, so the properties of this
   model will all be set to the default values.

1. Using all the feature variables, fit a the model to predict `Balance`.
   This can be done like this:

   ```python
   model.fit(X, y)
   ```

1. Take a look at the model's beta coefficients (`model.intercept_` and `model.coef_`).
   How can we interpret these numbers in terms of each feature's influence on predictions?

1. Predict all targets and call the result `y_hat`. Use the SciKit Learn documentation
   to see how to do this.

1. Look at a few values of `y` and `y_hat`. Do the predictions look reasonable?

1. Make a plot of predicted vs. actual values for a simple linear regression (one that
   has only one feature). The plot should show scattered dots for actual values and
   a line that represents predictions. Observe how the model has generated a line
   that is the "best fit". Here is some code that will do this. Note that we are using
   "Limit" as the one feature. Try choosing a diffeent feature and re-running the code:

   ```python
   simple_model = LinearRegression()

   X_limit_only = X[['Limit']]
   simple_model.fit(X_limit_only, y)
   y_hat_limit_only = simple_model.predict(X_limit_only)
   plt.scatter(X_limit_only, y, color='black')
   plt.plot(X_limit_only, y_hat_limit_only, color='blue', linewidth=3)

   plt.xticks(())
   plt.yticks(())

   plt.show()
   ```

1. Calcuate the R2 (R squared) score. Use the SciKit Learn documentation to see how
   to do this. Is it a good score? What units is it in?

1. Calculate the "root mean squared error". Again, consult the documenatation.
   How does it look? What units is it in?

1. Take a look again at your scatter matrix. Are there any redundant or
   colinear features? Do some features appear more linear than others?

1. Using the assumptions required of the linear regression model, are there features
   that could or should be eliminated? Try experimenting with different feature sets.
   How did this affect your scores?

## Extra Credit (Note: you will likely need to do Web searches to help you)

1. In the first section above, we ignored non-numeric features. These can
   still be used, however. One word of warning: you need to expect inconsistencies
   in the data. For example, some strings in the dataset have extra white
   space. Keep an eye out for this and be sure to handle it.

   For booleans, converting strings to 0/1 values can
   be done in Pandas (manipulating the dataframe). Web searchs will help here.

   But here is some starter code to convert boolean strings to 0 or 1:

   ```python
   # Grab the boolean feature columns
   df_booleans = df_full[['Gender', 'Married', 'Student']]

   # Strip whitespace off of string values, since dataset is inconsistent about this
   df_booleans = df_booleans.applymap(lambda x: x.strip() if type(x)==str else x)

   # Replace string values to 0 or 1
   df_booleans = df_booleans.replace({'No': 0, 'Yes': 1})
   df_booleans = df_booleans.replace({'Male': 0, 'Female': 1})
   ```
   For categorical features, take a look at SciKit Learn's "OneHotEncoder".

   Here's some code that works with the categorical feature, splitting it into several
   columns that contain combinations of integer values that linear models can handle:

   ```python
   # Grab the categorical feature column
   df_categorical = df_full[['Ethnicity']]

   # Strip whitespace off of string values, since dataset is inconsistent about this
   df_categorical = df_categorical.applymap(lambda x: x.strip() if type(x)==str else x)

   # Make new all-numeric dataframe
   drop_enc = OneHotEncoder(drop='first', sparse=False, dtype=np.int)
   onehot_array = drop_enc.fit_transform(df_categorical)
   df_onehot = pd.DataFrame(onehot_array, columns=['Ethnicity_1', 'Ethnicity_2'])
   ```

   Once you make the non-numeric features numeric, repeat the steps to do a
   Linear Regression using the new all-numeric dataframe including the new features.
   See if you can get better scores by including these features.
   Note: make sure you have removed the old non-numeric versions of these columns!
   (One way is to re-construct the dataframe from several dataframes only containing
   what you need. Using `pd.concat([[df1, df2,...], axis=1])` is a way to put the pieces back together.

1. Let's make another scatter plot showing only the actual values of one feature.
   Use "Limit" as the feature for this. Here's some code that will do this:

   ```python
   df.plot(kind='scatter', y='Balance', x='Limit', edgecolor='none', figsize=(12, 5))
   ```

   Do you notice some odd behavior around the zero balance value?
   Why do you think there is a clustering of data points around zero balance?

1. Because of these data points, the data is not as linear as we would like.
   Try plotting other features (insteat of "Limit") and see if they do not
   have this issue. These will be features that can differentiate most zero
   balance observations from non-zero balance observations. Try adding these
   and see if your scores improve.

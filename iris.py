'''


classification example using GaussianNB on the iris datasets   

    dataset:
    
        source: https://www.kaggle.com/datasets/uciml/iris    
        Class:  {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        
        
    training
        80/20 split
        GaussianNB
        random_state=42

    accuracy : 96-97%
    F1 Scores per class: [1.         0.9047619  0.89473684]
    
    
'''

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


########################################################
df = pd.read_csv("iris.csv")

df.describe()
df.info()

df = df.drop("Id", axis=1)
species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['species_numeric'] = df['Species'].map(species_map)
df = df.drop("Species", axis=1)




# keep only most correlated columns
df_not_string = df.select_dtypes(exclude=['object'])
df_corr = df_not_string.corr()
print(df_corr)

THRESHOLD = 0.50  ### this removes SepalWidthCm
corr_cols = df_corr.columns[abs(df_corr["species_numeric"]) >= THRESHOLD]
df2 = df[corr_cols]



########################################################
# Separate features (X) and target (y)
# Split data
X = df2.drop('species_numeric', axis=1)
#print(X.head())
y = df2['species_numeric']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.info())






########################################################
# Test


# Initialize and fit the GaussianNB classifier
clf = GaussianNB()
clf.fit(X_train, y_train)


# cross validation - to estimate stability and speed
from sklearn.model_selection import cross_validate
cv_results = cross_validate(clf, X_train, y_train, cv=3, scoring=('f1_macro'), return_estimator=True)
print(f"cv_results: {cv_results}")


# accuracy
score = clf.score(X_test, y_test)
print(f"score: {score}")


# mislabeled
y_pred = clf.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
      
# f1 score      
scores_per_class = f1_score(y_test, y_pred, average=None)
print(f"F1 Scores per class: {scores_per_class}")


# summary report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, zero_division=1))


'''


example using the car_kick datasets   
    https://www.kaggle.com/datasets/ulrikthygepedersen/car-kick/data
    
    
    Make and TopThreeAmericanName are one-hot encoded
    all string columns are not considered
    
    then only the columns that corrolates with "Class" with an absolute value of 0.11 or greater are kept
    
    training
        80/20 split
        GaussianNB
        random_state=42

    score = +- 86%
    
'''

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

########################################################
df = pd.read_csv("car_kick.csv")
#df.describe()
#df.info()
df = pd.get_dummies(df, columns=['Make'])
df = pd.get_dummies(df, columns=['TopThreeAmericanName'])



# keep only most correlated columns
df_not_string = df.select_dtypes(exclude=['object'])
df_corr = df_not_string.corr()
print(df_corr)
THRESHOLD = 0.11
corr_cols = df_corr.columns[abs(df_corr["Class"] ) >= THRESHOLD]
df2 = df[corr_cols]



########################################################
# Separate features (X) and target (y)
# Split data
X = df2.drop('Class', axis=1)
x_cols = X.columns
#print(X.head())
y = df2['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.info())


########################################################
# Initialize and fit the GaussianNB classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

########################################################
score = clf.score(X_test, y_test)
print(f"score: {score}")


'''
	test : preprocessing dataset with different types of data
'''

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Sample data
data = {
    'Age': [25.5, 30.0],
    'Has_Car': [True, False],
    'Color': ['Red', 'Blue'],
    'Income': [50000.0, 60000.0],
    'Target': ['Yes', 'No']
}
df = pd.DataFrame(data)

# Convert boolean to 0/1
df['Has_Car'] = df['Has_Car'].astype(int)

# One-hot encode categorical
df = pd.get_dummies(df, columns=['Color'])


print(df.describe())
print(df.head())
##############################


# Separate features (X) and target (y)
X = df.drop('Target', axis=1)
x_cols = X.columns
#print(X.describe())
#print(X.head())

y = df['Target']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize and fit the GaussianNB classifier
clf = GaussianNB()
clf.fit(X, y_encoded)


'''
    Example prediction
    
        - sklearn will issue a werning about the column names, if they're not the same as during the training
        
            "X does not have valid feature names, but GaussianNB was fitted with feature names"

             instead of just an array:      
                new_sample = [[28.0, 1, 55000, 0, 1]]  # Age, Has_Car, Income, Color_Blue, Color_Red
                
             we use a dataframe with the same columns names
             
         - Important:
         
            Columns must be in the same order as they appear in the training dataset.
         
            Remember that using "get_dummies" to one-hot encode the colors had the effect of making 
                the "Income" column appear before the new color columns.
                What's more, the order of the one-hot ecncoded columns does not necessarily match the way the values appeared in the dataset
                
                the best is to use X.columns to construct the sample

'''
new_sample = pd.DataFrame([[28.0, 1, 55000.0, 0, 1]], columns=x_cols )
print(new_sample.head())


predicted_label = clf.predict(new_sample)
predicted_class = label_encoder.inverse_transform(predicted_label)

print(f"Predicted class: {predicted_class[0]}")

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

'''
Dataset: [Outlook, Temp, Humidity, Windy]
    Outlook: 0=Overcast, 1=Rainy, 2=Sunny
    Temp: 0=Cool, 1=Hot, 2=Mild
    Humidity: 0=High, 1=Normal
    Windy: 0=False, 1=True
'''
X = np.array([
    [2, 1, 0, 0], # Sunny, Hot, High, False
    [2, 1, 0, 1], # Sunny, Hot, High, True
    [0, 1, 0, 0], # Overcast, Hot, High, False
    [1, 2, 0, 0], # Rainy, Mild, High, False
    [1, 0, 1, 0], # Rainy, Cool, Normal, False
    [0, 0, 1, 1], # Overcast, Cool, Normal, True
    [2, 2, 0, 0]  # Sunny, Mild, High, False
])

# Labels: 0 = No Play, 1 = Play
y = np.array([0, 0, 1, 1, 1, 1, 0])

# Initialize and Train CategoricalNB
# We use CategoricalNB because our features are discrete categories
model = CategoricalNB()
model.fit(X, y)

# Predict for a new day: [Sunny, Cool, Normal, True]
new_day = np.array([[2, 0, 1, 1]])
prediction = model.predict(new_day)
probability = model.predict_proba(new_day)

'''
    should output:
        Should we play golf? Yes
        Probabilities: [No: 0.40, Yes: 0.60]
'''        
print(f"Should we play golf? {'Yes' if prediction[0] == 1 else 'No'}")
print(
    f"Probabilities: [No: {probability[0][0]:.2f}, Yes: {probability[0][1]:.2f}]"
)

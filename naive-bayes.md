# Naive Bayes

---

## Section 1 - What is Naive Bayes?

**Imagine you’re a doctor.** A patient walks in with a fever, a cough, and a headache. You need to diagnose whether they have the flu, a cold, or just allergies. How do you decide? You might think: “Given these symptoms, what’s the most likely disease?”

Naive Bayes works in a similar way. It’s a simple but powerful algorithm for classification : the task of assigning a label (like “flu” or “cold”) to an input (like symptoms). The “naive” part comes from its core assumption: **it treats each feature (or symptom) as independent of the others**, even if that’s not always true in reality. This makes calculations much faster and often works surprisingly well, especially with text or high-dimensional data.

### Why Use Naive Bayes?

- **Fast and efficient**: It requires less data and computational power than many other algorithms.
- **Works well with text**: It’s a go-to for tasks like spam detection, sentiment analysis, and document categorization.
- **Handles high dimensions**: Even with thousands of features, Naive Bayes remains practical.

### Real-World Example: Spam Detection

Suppose you receive an email with the words “free,” “win,” and “prize.” Naive Bayes would calculate the probability that this email is spam, given those words, assuming the presence of each word doesn’t affect the others. If the probability is high enough, it labels the email as spam.

---

## Section 2 - How Naive Bayes Works

### Bayes’ Theorem: The Foundation

At its heart, Naive Bayes is based on **Bayes’ Theorem**, which describes how to update the probability of a hypothesis (e.g., “this email is spam”) as we see more evidence (e.g., words like “free” or “win”).  In this context, the theorem becomes:

$$
P(\text{Spam} \mid \text{Free, Win}) = \frac{P(\text{Free, Win} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{Free, Win})}
$$

**Probability of the email is spam given the words “free” and “win.”**

$$
P(Spam∣Free, Win)
$$

> This is what we are trying to estimate.



**Probability of seeing “free” and “win” in any email.**

$$
P(Free, Win)
$$

> How to compute it ? This is simply the number of emails that contain the words "free" and "win" divided by the number of all emails. 



**Probability of seeing “free” and “win” in spam emails.**

$$
P(Free, Win∣Spam)
$$

> How to compute it ?  This is the number of emails that contain the words "free" and "win" divided by the number of emails that have been classified as "Spam" so far.



**Overall probability any email is spam.**

$$
P(Spam)
$$

> This the number of "spam" divided by the overall number of emails received.



### The “Naive” Assumption

Calculating P(Free, Win∣Spam) directly is complex, especially with many features. Naive Bayes simplifies this by assuming **all features are independent**:

$$
P(Free, Win∣Spam)=P(Free∣Spam)⋅P(Win∣Spam)
$$

This “naive” assumption makes the math tractable, even if it’s not always realistic.

### Types of Naive Bayes

Naive Bayes comes in different “flavors,” each tailored to a specific type of data: **Gaussian** for continuous numbers, **Multinomial** for discrete counts (like word frequencies, like in our spam example), and **Bernoulli** for binary features (like presence/absence of words).

**Naive Bayes Variants**

| Variant     | Use Case                     | Example Data Type             |
| ----------- | ---------------------------- | ----------------------------- |
| Gaussian    | Continuous numerical data    | Height, weight, temperature   |
| Multinomial | Discrete counts (e.g., text) | Word frequencies in documents |
| Bernoulli   | Binary features              | Presence/absence of words     |

---

## 3 - Naive Bayes in Python with scikit-learn

Using Naive Bayes for spam detection is a classic "hello world" of machine learning. It's particularly effective for text because it handles high-dimensional data (like a vocabulary of thousands of words) surprisingly well, even with its "naive" assumption that every word is independent of the others.

The table below shows which Scikit-Learn class is usually preferred based on the **mathematical nature** of their input features X

| **Scikit-Learn Class** | **Feature Type**     | **Mathematical Assumption**                              | **Real-World Example**                                           |
| ---------------------- | -------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| **GaussianNB**         | **Continuous**       | Data follows a Normal (Bell Curve) distribution.         | Sensor readings (Temp, Voltage), Human height/weight.            |
| **MultinomialNB**      | **Discrete Counts**  | Features represent frequencies or tallies.               | Word counts in a document, number of clicks on a button.         |
| **BernoulliNB**        | **Binary / Boolean** | Features are independent "Yes/No" or "1/0" trials.       | Presence/Absence of a specific keyword, "Is the user logged in?" |
| **CategoricalNB**      | **Categorical**      | Features are discrete categories with no inherent order. | Eye color (Blue, Brown, Green), Job title, Country of origin.    |


Here is a short example of a `MultinomialNB` classifier that uses`CountVectorizer`. 

**Note**: *CountVectorizer* turns the sentences into a "Bag of Words." It counts how many times "URGENT" or "lunch" appears, creating a numerical matrix the model can understand.

### Full Code Example

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Data
emails = [
    "Hey, are we still meeting for lunch today?",
    "CONGRATULATIONS! You've won a $1,000 Walmart gift card. Click here.",
    "Can you send me the report by 5 PM?",
    "URGENT: Your account has been compromised. Verify now for cash.",
    "Don't forget to pick up milk on your way home.",
    "FREE entry to our annual prize draw! Text WIN to 80085"
]
# 0 = Not Spam (Ham), 1 = Spam
labels = [0, 1, 0, 1, 0, 1]

# Vectorize the text (Convert words to numbers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data (Normally you'd use a larger dataset)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on new data
new_emails = ["Get a free gift card!", "Are you home?"]
new_emails_counts = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_counts)

print(f"Predictions: {predictions}") # 1 is Spam, 0 is Not Spam
```

Here's another example using a well known dataset.

```python
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

print(f"Should we play golf? {'Yes' if prediction[0] == 1 else 'No'}")
print(
    f"Probabilities: [No: {probability[0][0]:.2f}, Yes: {probability[0][1]:.2f}]"
)
```

---

## Section 4 - When to Use (and Not Use) Naive Bayes

Naive Bayes is the "efficient specialist" of machine learning. It trades off complex nuance for extreme speed and reliability in specific contexts.

### Strengths: Where it Excels

- **Computational Speed:** Training involves simple counting rather than iterative optimization. It is often orders of magnitude faster than Logistic Regression or Support Vector Machines (SVM).

- **High-Dimensional Data:** It is the "King of Text." In spam filtering, where every word is a feature, Naive Bayes handles thousands of dimensions without the "curse of dimensionality" that slows down other models.

- **Minimal Data Requirements:** Because it doesn't try to learn complex relationships between features, it reaches peak performance with significantly less training data than more "sophisticated" algorithms.

### Limitations: The "Naive" Reality

The algorithm’s primary weakness is its **Independence Assumption**: it assumes the presence of one word is entirely unrelated to another.

- **Context Blindness:** In reality, words are highly dependent. If a message contains "Nigerian," it is statistically likely to also contain "Prince." Naive Bayes ignores this link, treating every word as a solo actor.

- **The Zero-Frequency Problem:** If the model encounters a word in the test set that it never saw during training, the probability drops to zero, breaking the entire calculation. This requires **Laplace Smoothing** (adding a small constant to all counts) to fix.

- **Poor Probability Estimates:** While great at picking the right *category*, it is notoriously bad at providing the actual *probability*. It tends to be overconfident, outputting values very close to 0 or 1.

### Comparison at a Glance

| **Algorithm**           | **Best Used When...**                | **Downside vs. Naive Bayes**                     |
| ----------------------- | ------------------------------------ | ------------------------------------------------ |
| **Logistic Regression** | You need precise probability scores. | Slower; prone to overfitting on small text sets. |
| **SVM**                 | You need maximum accuracy on text.   | High memory usage; harder to interpret.          |
| **Random Forest**       | Feature interactions are critical.   | Requires much more data and tuning.              |

> **Takeaway:** Use Naive Bayes as a **baseline**. It provides a "good enough" solution quickly, allowing you to decide if a more complex model is worth the extra effort.

---

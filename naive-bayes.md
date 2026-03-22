# Naive Bayes

---

## Section 1 - What is Naive Bayes?

**Imagine a gardener** who finds that one of its favorite plants in his garden has yellow leaves, wilting stems, and spots on its foliage. To determine whether the plant is suffering from overwatering, a fungal infection, or a nutrient deficiency, he ask himself: “Given these signs, what is the most likely cause?”
Naive Bayes works in a similar way. It is a simple yet powerful algorithm for classification, i.e. the process of assigning a label (such as “overwatering” or “fungal infection”) to an input (such as plant symptoms). The term “naive” comes from its core assumption: **the algorithm treats each feature (or sign) as independent of the others**, even if they might be related in reality. This simplification makes calculations faster and often produces effective results, especially with text or complex datasets.


### Why Use Naive Bayes?

- **Fast and efficient**: It requires less data and computational power than many other algorithms.
- **Works well with text**: It’s a go-to for tasks like spam detection, sentiment analysis, and document categorization.
- **Handles high dimensions**: Even with thousands of features, Naive Bayes remains practical.

### Real-World Example: Spam Detection

Suppose you receive an email with the words “free,” “win,” and “prize.” Naive Bayes would calculate the probability that this email is spam, given those words, assuming the presence of each word doesn’t affect the others. If the probability is high enough, it labels the email as spam.

---

## Section 2 - How Naive Bayes Works

### Bayes’ Theorem: The Foundation

At its heart, Naive Bayes is based on **Bayes’ Theorem**,

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

The following arrangement highlights that Bayes’ Theorem is a dynamic belief update:

$$
P(A|B) = \frac{P(B|A)}{P(B)} \cdot P(A)
$$

---

* **$P(A)$ (The Prior): Our baseline belief.**
  This is a "pure" probability that does not consider any specific conditions, constraints, or new evidence (e.g., the general prevalence of the flu in the entire population, or the proportion of spams among all emails received).

* **$\frac{P(B|A)}{P(B)}$ (The Evidence Ratio): The "Scaling Factor."**
  This adjusts the baseline based on how the new evidence ($B$) relates to the outcome ($A$):
  
  * **Greater than 1:** The evidence is more likely to occur with this outcome than by chance; our confidence increases.
  
  * **Equal to 1:** The evidence is neutral; our confidence remains the same.
  
  * **Less than 1:**  The evidence is rare for this outcome but common otherwise; our confidence decreases.

---

In this context of our spam example, the equation could be:

$$
P(\text{Spam} \mid \text{Free}) = \frac{P(\text{Free} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{Free})}
$$

which describes how to update the probability of the hypothesis (e.g., “this email is spam”) as we see more evidence (e.g., words like “free” or “win”). Here's the meaning of differents parts in the equation :

---

**Probability of the email is spam given the words “free”**

$$
P(Spam∣Free)
$$

> This is what we are trying to estimate.



**Probability of seeing “free” in any email.**

$$
P(Free)
$$

> How to compute it ? This is simply the number of emails that contain the words "free" divided by the number of all emails. 



**Probability of seeing “free” in spam emails.**

$$
P(Free ∣ Spam)
$$

> How to compute it ?  this is the proportion of known spam emails that contain the word "free". In order words, it is the number of spams with the word "free", **divided by** the total number of spams in the mailbox.



**Overall probability any email is spam.**

$$
P(Spam)
$$

> This the number of spams divided by the overall number of emails received.

#### How the Maths work

In order to facilitate the calculations, two tables are often used to pre-compute intermediate values: the *frequency table* and the *likelyhood table*.

Let's suppose we have a thousand emails, 200 of which were previously identified as spam. The word "free" appears in 140 of them, but also appears in 10 legitimate emails. Here's how we compute the "frequency table" for this single feature example:


**Frequency table**

| Class|Contains "Free"|Does Not Contain "Free"|Total
|--|--|--|--|
|Spam|140|60|200|
|Not Spam|10|790|800|
|Total|150|850|1000|
|Probability|150/1000 = **0.15**|850/1000 = 0,85| 1000 |


**Likelyhood table**

|Class|Likelihood of "Free"|Likelihood of Not "Free"|Prior Probability|
|--|--|--|--|
|Spam|140/200=**0.70**|60/200=0.30|200/1000=**0.20**|
|Not Spam|10/800=0.0125|790/800=0.9875|800/1000=0.80|

Since $P(\text{"Free"} \mid \text{Spam})$ is $0.70$ and $P(\text{"Free"} \mid \text{Not Spam})$ is only $0.0125$, the word "free" is a strong indicator of spam.


To classify a new incoming email using Naive Bayes (actually, it's only "Bayes", without the "Naive" part, since we only have one feature), we calculate the probability that it belongs to each class (Spam vs. Not Spam) and then choose the one with the higher value. Using the values from our frequency table and likelihood table:

**For the Spam Class:**
* $P(\text{"Free"} \mid \text{Spam}) = 0.70$
* $P(\text{Spam}) = 0.20$
* $P(\text{"Free"}) = 0.15$
* **Calculation:** $0.70 \times 0.20 / 0.15 = \mathbf{0.93}$

**For the Not Spam Class:**
* $P(\text{"Free"} \mid \text{Not Spam}) = 0.125$
* $P(\text{Not Spam}) = 0.80$
* $P(\text{Not "Free"}) = 0.85$
* **Calculation:** $0.0125 \times 0.80 / 0.85 = \mathbf{0.012}$

The new email could be flagged as a potential spam, since or 0.93 is way higher than 0.012.


---

### The “Naive” Assumption

With two of more features, the calculation quickly becomes complicated and difficult. To simplify the calculation of $P(\text{Free}, \text{Win} \mid \text{Spam})$, Naive Bayes avoids the complexity of analyzing word combinations by **assuming all features are entirely independent**. Under this "naive" assumption, the probability of multiple features occurring together is reduced to a simple product of their individual probabilities:

$$
P(Free, Win∣Spam)=P(Free∣Spam)⋅P(Win∣Spam)
$$

This “naive” assumption makes the math tractable, even if it’s not always realistic.



See these pages for more details about the maths behind Naive Bayes:
- [Kaggle](https://www.kaggle.com/code/pavelbogdanov/spam-filtering-with-naive-bayes)
- [Geeks For Geeks](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/)
- [Scikit Learn](https://scikit-learn.org/stable/modules/naive_bayes.html)


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

**Note**: *CountVectorizer* turns the sentences into a "Bag of Words." For example, it counts how many times "URGENT" or "lunch" appears, creating a numerical matrix the model can understand.

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

Here's another example using a well known dataset. This time, we use the class CategoricalNB; our features are *discrete categories with no inherent order*.

```python
from sklearn.naive_bayes import CategoricalNB
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

### Common Use Cases

While modern deep learning (like Transformers) dominates the spotlight, Naive Bayes remains a production staple in 2026 because it is incredibly fast, memory-efficient, and works well with limited data. Here are some the most common use cases:

| Use Case | How it's Applied | Why Naive Bayes is chosen |
| :--- | :--- | :--- |
| **Email Filtering** | Real-time classification of incoming mail into "Spam," "Promotions," or "Primary." | Lightning-fast inference allows providers to scan billions of emails instantly. |
| **Sentiment Analysis** | Analyzing social media "live feeds" or customer reviews to tag them as Positive/Negative. | Highly effective at handling "Bag of Words" data where word order is less critical than presence. |
| **Medical Diagnosis** | Early-stage screening based on discrete symptoms (e.g., fever, cough, fatigue). | It handles categorical clinical data well and provides a clear probability "score" for doctors. |
| **News Categorization** | Automatically tagging articles into "Sports," "Politics," or "Tech" taxonomies. | Excels in multi-class problems and scales linearly as the number of topics grows. |
| **Fraud Detection** | Identifying unusual spending patterns or suspicious login locations. | Its speed makes it ideal for "edge" cases where a transaction must be approved or flagged in milliseconds. |
| **Resource Recommendation** | Suggesting related articles or products in low-latency environments. | Often used as a "baseline" component because it requires very little training data to be effective. |



### Limitations: The "Naive" Reality

The algorithm’s primary weakness is its **Independence Assumption**: it assumes the presence of one word is entirely unrelated to another.

- **Context Blindness:** In reality, words are highly dependent. If a message contains "Nigerian," it is statistically likely to also contain "Prince." Naive Bayes ignores this link, treating every word as a solo actor.

- **The Zero-Frequency Problem:** If the model encounters a word in the test set that it never saw during training, the probability drops to zero, breaking the entire calculation. This requires **Laplace Smoothing** (adding a small constant to all counts) to fix.

- **Poor Probability Estimates:** While great at picking the right *category*, it is notoriously bad at providing the actual *probability*. It tends to be overconfident, outputting values very close to 0 or 1.

---


### The **Zero-Frequency Problem**

In Naive Bayes, if a word (like "Extravaganza") has **zero** occurrences in the training data for a specific category, the probability for that entire category becomes **zero** due to multiplication ($0 \times \text{everything else} = 0$). This is known as the **Zero-Frequency Problem**.


#### Laplace Smoothing: The "+1" Rule
To prevent a single "unseen" word from crashing the model, we add a small "hallucinated" count to every word.

1.  **Add 1** to every raw count in the frequency table.
2.  **Add the Vocabulary Size** ($V$) to the denominator when calculating likelihoods to keep the total probability equal to 1.

$$
P(\text{Word} \mid \text{Class}) = \frac{\text{Count}(\text{Word, Class}) + 1}{\text{Total Words in Class} + V}
$$


##### Why use it?
* **Safety Net:** Ensures no probability is ever exactly zero.
* **Stability:** Makes the model robust against small datasets or rare words.
* **Simplicity:** It's a computationally "cheap" way to handle missing data.

[Additive Smoothing on Wikipedia](https://en.wikipedia.org/wiki/Additive_smoothing)

---



### Comparison with Other Classification Algorythms

| **Algorithm**           | **Best Used When...**                | **Downside vs. Naive Bayes**                     |
| ----------------------- | ------------------------------------ | ------------------------------------------------ |
| **Logistic Regression** | You need precise probability scores. | Slower; prone to overfitting on small text sets. |
| **SVM**                 | You need maximum accuracy on text.   | High memory usage; harder to interpret.          |
| **Random Forest**       | Feature interactions are critical.   | Requires much more data and tuning.              |

> **Takeaway:** Use Naive Bayes as a **baseline**. It provides a "good enough" solution quickly, allowing you to decide if a more complex model is worth the extra effort.

---

[More info on SkLearn](https://scikit-learn.org/stable/modules/naive_bayes.html)


---
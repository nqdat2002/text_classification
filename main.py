from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dữ liệu mẫu
texts = [
    "Win a free iPhone now", 
    "Meeting at 10am", 
    "Congratulations, you won a lottery", 
    "Let's have lunch tomorrow"
]
labels = [1, 0, 1, 0] 

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

test_text = ["Free lottery tickets", "See you at the office"]
X_test = vectorizer.transform(test_text)
predictions = model.predict(X_test)
print(predictions) 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import glob

def read_file(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [(line.strip(), label) for line in lines]

data = []
data += read_file('spam.txt', 'spam')
data += read_file('not_spam.txt', 'not_spam')
data += read_file('positive.txt', 'positive')
data += read_file('negative.txt', 'negative')
data += read_file('sports.txt', 'sports')
data += read_file('politics.txt', 'politics')

texts, labels = zip(*data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

test_texts = [
    "Bạn đã trúng thưởng xe máy.",
    "Dịch vụ rất tốt, tôi hài lòng.",
    "Đội tuyển bóng đá Việt Nam vô địch.",
    "Chính phủ vừa ban hành chính sách mới."
]
X_test = vectorizer.transform(test_texts)
predictions = model.predict(X_test)

for text, label in zip(test_texts, predictions):
    print(f'"{text}" => {label}')
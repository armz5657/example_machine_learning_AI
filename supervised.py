from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# ตัวอย่างข้อความอีเมล
emails = ["Win a free iPhone now", "Meeting at 5 PM", "You won a lottery!", "Project deadline is tomorrow"]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# แปลงข้อความเป็นตัวเลข
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# แบ่งข้อมูลออกเป็นชุดฝึก (train) และทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# สร้างและฝึกโมเดล Naïve Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# ทดสอบโมเดลด้วยข้อมูลใหม่
print("Prediction:", model.predict(X_test))

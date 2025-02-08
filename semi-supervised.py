import numpy as np
from sklearn.semi_supervised import LabelSpreading

# ข้อมูลตัวอย่าง (พฤติกรรมของลูกค้า)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# มีป้ายกำกับบางส่วน (-1 คือไม่มีป้ายกำกับ)
y = np.array([0, 1, -1, -1, 1, 0])

# ใช้ Label Spreading ในการเรียนรู้จากข้อมูลที่มี Label และเติมค่าที่ขาด
model = LabelSpreading(kernel='knn', n_neighbors=2)
model.fit(X, y)

# แสดงป้ายกำกับที่ถูกเติมเข้ามา
print("Predicted Labels:", model.transduction_)
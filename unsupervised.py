import numpy as np
from sklearn.cluster import KMeans

# สร้างข้อมูลตัวอย่าง (2 มิติ)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# ใช้ K-Means แบ่งเป็น 2 กลุ่ม
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# แสดงผลลัพธ์ของการจัดกลุ่ม
print("Cluster Centers:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)

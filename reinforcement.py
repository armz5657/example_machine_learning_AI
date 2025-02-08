import numpy as np

# กำหนดสถานะและรางวัลใน Environment (ตัวอย่าง Q-learning)
states = ["A", "B", "C"]
actions = ["left", "right"]
rewards = {
    ("A", "right"): 1,
    ("B", "left"): 1,
    ("B", "right"): -1,
    ("C", "left"): -1
}

# Q-Table
Q = np.zeros((len(states), len(actions)))

# อัปเดตค่า Q-learning
for _ in range(100):
    state = np.random.choice(states)  # สุ่มสถานะ
    action = np.random.choice(actions)  # สุ่มการกระทำ
    reward = rewards.get((state, action), 0)  # ดึงค่ารางวัล
    Q[states.index(state)][actions.index(action)] += reward  # อัปเดตค่า Q-table

print("Q-Table:", Q)

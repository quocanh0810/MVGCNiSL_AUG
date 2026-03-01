import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng K562
k562_data = {
    "Input Network": [
        "K562",
        "K562_TransE",
        "K562_AUG-40",
        "K562_AUG-80",
        "K562_AUG-100"
    ],
    "AUC": [0.8289, 0.8394, 0.8428, 0.8405, 0.7582],
    "AUPR": [0.8183, 0.8286, 0.8332, 0.8298, 0.7534],
    "P@5": [0.9032, 0.9113, 0.9436, 0.9113, 0.9032],
    "P@10": [0.9032, 0.8992, 0.9113, 0.9154, 0.8831]
}

# Chuyển đổi dữ liệu thành DataFrame
df_k562 = pd.DataFrame(k562_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_k562.set_index("Input Network").plot(kind="bar", figsize=(12, 6))

# Cấu hình biểu đồ
plt.title("K562 + 4 network", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

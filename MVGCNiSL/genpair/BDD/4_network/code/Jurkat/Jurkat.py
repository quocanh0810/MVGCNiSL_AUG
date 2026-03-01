import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng Jurkat
jurkat_data = {
    "Input Network": [
        "Jurkat",
        "Jurkat_TransE",
        "Jurkat_AUG-40",
        "Jurkat_AUG-80",
        "Jurkat_AUG-100"
    ],
    "AUC": [0.7889, 0.8038, 0.8248, 0.8259, 0.7564],
    "AUPR": [0.7837, 0.8218, 0.8339, 0.8379, 0.7681],
    "P@5": [0.8333, 1.0000, 1.0000, 1.0000, 0.8333],
    "P@10": [0.9167, 0.9375, 1.0000, 0.9375, 0.9167]
}

# Chuyển đổi dữ liệu thành DataFrame
df_jurkat = pd.DataFrame(jurkat_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_jurkat.set_index("Input Network").plot(kind="bar", figsize=(12, 6))

# Cấu hình biểu đồ
plt.title("Jurkat + 4 network", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

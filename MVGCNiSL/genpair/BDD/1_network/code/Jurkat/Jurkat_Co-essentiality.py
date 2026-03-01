import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng
jurkat_coessentiality_data = {
    "Input Network": [
        "Jurkat_Coess",
        "Jurkat_AUG-40_Coess",
        "Jurkat_AUG-80_Coess",
        "Jurkat_AUG-100_Coess"
    ],
    "AUC": [0.8103, 0.8114, 0.8001, 0.6592],
    "AUPR": [0.8299, 0.8018, 0.8098, 0.6858],
    "P@5": [1.0000, 1.0000, 1.0000, 0.8333],
    "P@10": [1.0000, 0.9194, 1.0000, 0.9167]
}

# Chuyển dữ liệu thành DataFrame
df_jurkat_coessentiality = pd.DataFrame(jurkat_coessentiality_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_jurkat_coessentiality.set_index("Input Network").plot(kind="bar", figsize=(12, 6))
plt.title("Jurkat Co-essentiality", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()
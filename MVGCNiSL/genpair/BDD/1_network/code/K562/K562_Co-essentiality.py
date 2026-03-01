import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng
k562_coessentiality_data = {
    "Input Network": [
        "K562_Coess",
        "K562_AUG-40_Coess",
        "K562_AUG-80_Coess",
        "K562_AUG-100_Coess"
    ],
    "AUC": [0.7973, 0.8114, 0.8059, 0.4982],
    "AUPR": [0.7952, 0.8018, 0.8013, 0.5071],
    "P@5": [0.9355, 1.0000, 0.9355, 0.5806],
    "P@10": [0.9194, 0.9194, 0.9194, 0.5484]
}

# Chuyển dữ liệu thành DataFrame
df_k562_coessentiality = pd.DataFrame(k562_coessentiality_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_k562_coessentiality.set_index("Input Network").plot(kind="bar", figsize=(12, 6))
plt.title("K562 Co-essentiality", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

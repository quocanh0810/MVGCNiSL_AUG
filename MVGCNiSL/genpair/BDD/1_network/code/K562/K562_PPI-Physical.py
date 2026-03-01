import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng
k562_ppi_physical_data = {
    "Input Network": [
        "K562_Physical",
        "K562_AUG-40_Physical",
        "K562_AUG-80_Physical",
        "K562_AUG-100_Physical"
    ],
    "AUC": [0.7136, 0.8232, 0.8347, 0.5448],
    "AUPR": [0.7316, 0.8143, 0.8257, 0.5428],
    "P@5": [0.9355, 0.9677, 0.9677, 0.6774],
    "P@10": [0.8710, 0.9194, 0.9355, 0.5968]
}

# Chuyển dữ liệu thành DataFrame
df_k562_ppi_physical = pd.DataFrame(k562_ppi_physical_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_k562_ppi_physical.set_index("Input Network").plot(kind="bar", figsize=(12, 6))
plt.title("K562 PPI-Physical", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

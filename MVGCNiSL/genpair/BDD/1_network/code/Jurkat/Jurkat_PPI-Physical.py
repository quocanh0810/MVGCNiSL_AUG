import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng
jurkat_ppi_physical_data = {
    "Input Network": [
        "Jurkat_Physical",
        "Jurkat_AUG-40_Physical",
        "Jurkat_AUG-80_Physical",
        "Jurkat_AUG-100_Physical"
    ],
    "AUC": [0.7334, 0.7691, 0.7603, 0.5951],
    "AUPR": [0.7379, 0.8089, 0.7775, 0.6029],
    "P@5": [0.8333, 1.0000, 1.0000, 0.6667],
    "P@10": [0.9167, 1.0000, 0.9167, 0.7500]
}

# Chuyển dữ liệu thành DataFrame
df_jurkat_ppi_physical = pd.DataFrame(jurkat_ppi_physical_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_jurkat_ppi_physical.set_index("Input Network").plot(kind="bar", figsize=(12, 6))

# Cấu hình biểu đồ
plt.title("Jurkat PPI-Physical", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

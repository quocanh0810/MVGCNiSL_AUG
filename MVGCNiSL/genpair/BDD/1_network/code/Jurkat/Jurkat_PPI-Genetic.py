import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng
jurkat_ppi_genetic_data = {
    "Input Network": [
        "Jurkat_Genetic",
        "Jurkat_AUG-40_Genetic",
        "Jurkat_AUG-80_Genetic",
        "Jurkat_AUG-100_Genetic"
    ],
    "AUC": [0.7748, 0.8253, 0.8127, 0.7321],
    "AUPR": [0.7848, 0.8388, 0.8213, 0.7664],
    "P@5": [0.8333, 1.0000, 1.0000, 1.0000],
    "P@10": [0.9167, 0.9167, 0.9167, 1.0000]
}

# Chuyển dữ liệu thành DataFrame
df_jurkat_ppi_genetic = pd.DataFrame(jurkat_ppi_genetic_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_jurkat_ppi_genetic.set_index("Input Network").plot(kind="bar", figsize=(12, 6))
plt.title("Jurkat PPI-Genetic", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

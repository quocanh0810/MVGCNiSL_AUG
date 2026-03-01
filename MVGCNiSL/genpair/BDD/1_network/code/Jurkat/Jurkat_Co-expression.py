import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng Jurkat Co-expression
jurkat_co_expression_data = {
    "Input Network": [
        "Jurkat_Coexp",
        "Jurkat_AUG-40_Coexp",
        "Jurkat_AUG-80_Coexp",
        "Jurkat_AUG-100_Coexp"
    ],
    "AUC": [0.5493, 0.5582, 0.6332, 0.4636],
    "AUPR": [0.5620, 0.5943, 0.6577, 0.4762],
    "P@5": [0.6667, 0.8333, 0.8333, 0.5000],
    "P@10": [0.5833, 0.7500, 0.8333, 0.4167]
}

# Chuyển đổi dữ liệu thành DataFrame
df_jurkat_coexp = pd.DataFrame(jurkat_co_expression_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_jurkat_coexp.set_index("Input Network").plot(kind="bar", figsize=(12, 6))

# Cấu hình biểu đồ
plt.title("Jurkat Co-expression", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu từ bảng
k562_coexpression_data = {
    "Input Network": [
        "K562_Coexp",
        "K562_AUG-40_Coexp",
        "K562_AUG-80_Coexp",
        "K562_AUG-100_Coexp"
    ],
    "AUC": [0.6620, 0.7836, 0.7895, 0.5136],
    "AUPR": [0.6366, 0.7830, 0.7852, 0.5123],
    "P@5": [0.7419, 0.9677, 0.9355, 0.5806],
    "P@10": [0.6613, 0.8871, 0.8871, 0.5161]
}

# Chuyển dữ liệu thành DataFrame
df_k562_coexpression = pd.DataFrame(k562_coexpression_data)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = df_k562_coexpression.set_index("Input Network").plot(kind="bar", figsize=(12, 6))
plt.title("K562 Co-expression", fontsize=16, ha='center')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Input Network", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=10)  # Đặt góc quay bằng 0 và căn giữa chữ
plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.1, 1))  # Đặt thanh metrics ra ngoài biểu đồ
plt.tight_layout()  # Tránh bị cắt mất nội dung

# Hiển thị biểu đồ
plt.show()

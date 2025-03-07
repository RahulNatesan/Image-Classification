import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "data.csv"  
df = pd.read_csv(dataset_path)

print("Dataset Shape:", df.shape)

labels = df.iloc[:, 0]  
images = df.iloc[:, 1:].values  
images = images.reshape(-1, 28, 28)  

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    axes[i].imshow(images[i], cmap="gray")
    axes[i].set_title(f"Label: {labels[i]}")
    axes[i].axis("off")

plt.show()

print("Single image shape:", images[0].shape)

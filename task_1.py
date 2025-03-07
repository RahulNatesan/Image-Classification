import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

file_path = "data.csv"  
df = pd.read_csv(file_path)

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   

X = X / 255.0  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

def show_samples(n=5):
    indices = np.random.choice(len(X_test), n, replace=False)
    plt.figure(figsize=(n*2, 2))
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)  
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}")
        plt.axis("off")
    plt.show()

show_samples()

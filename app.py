
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# Upload Dataset
# ---------------------------
uploaded = files.upload()

# Get uploaded file name
file_name = list(uploaded.keys())[0]

# Load dataset
df = pd.read_csv(file_name)

print(df.head())

# ---------------------------
# Visualization
# ---------------------------

plt.figure()
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.show()

plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------
# Data Processing
# ---------------------------

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Model Training
# ---------------------------

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# Model Testing
# ---------------------------

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
     
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving archive (7).zip to archive (7) (2).zip
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \
0   52    1   0       125   212    0        1      168      0      1.0      2   
1   53    1   0       140   203    1        0      155      1      3.1      0   
2   70    1   0       145   174    0        1      125      1      2.6      0   
3   61    1   0       148   203    0        1      161      0      0.0      2   
4   62    0   0       138   294    1        1      106      0      1.9      1   

   ca  thal  target  
0   2     3       0  
1   0     3       0  
2   0     3       0  
3   1     3       0  
4   3     2       0  


Model Accuracy: 0.9853658536585366
Confusion Matrix:
[[102   0]
 [  3 100]]

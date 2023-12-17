import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Now that the file has been uploaded, let's proceed with loading the dataset and training the model
file_path = 'training data.csv'

# Loading the larger dataset
df = pd.read_csv(file_path)

# Encoding the categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == object or df[column].dtype == 'category':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Splitting the dataset
X = df.drop('Satisfaction', axis=1)
y = df['Satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Making predictions
y_pred = rf.predict(X_test)

# Calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy of the model:", accuracy)

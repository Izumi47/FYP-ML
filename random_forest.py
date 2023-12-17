import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
data = pd.read_csv('training data.csv')

# Preprocessing steps
data['CGPA'] = data['CGPA'].map({'Below 2.00': 0, '2.00 to 2.50': 1, '2.51 to 3.00': 2, '3.01 to 3.50': 3, '3.50 to 3.99': 4, '4': 5})
data['Class Performance [Bahasa Melayu]'] = data['Class Performance [Bahasa Melayu]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data['Class Performance [English]'] = data['Class Performance [English]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data['Class Performance [Sejarah]'] = data['Class Performance [Sejarah]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data['Class Performance [Mathematics]'] = data['Class Performance [Mathematics]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data['Class Performance [Science]'] = data['Class Performance [Science]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})

interests_list = ['Mathematics', 'Science (Physics, Chemistry, Biology)', 'Literature and Language Arts', 'History and Social Studies', 'Technology and Computer Science', 'Art and Creative Expression', 'Physical Education and Sports', 'Music and Performing Arts', 'Environmental Studies', 'Career and Vocational Interests']

for interest in interests_list:
    data['Interest_' + interest] = data['Interests'].apply(lambda x: 1 if interest in x else 0)
data.drop('Interests', axis=1, inplace=True)

data['Ambition'] = data['Ambition'].map({'Doctor or Healthcare Professional': 0, 'Engineer (e.g., Mechanical, Civil, Electrical)': 1, 'Computer Scientist or Software Developer': 2, 'Teacher or Educator': 3, 'Artist or Creative Professional': 4, 'Entrepreneur or Business Owner': 5, 'Musician or Performer': 6, 'Lawyer or Legal Professional': 7, 'Athlete or Sports Coach': 8, 'Scientist or Researcher': 8})

data['Current Class Stream'] = data['Current Class Stream'].map({'Science Stream': 0, 'Arts Stream': 1, 'Commerce Stream': 2, 'Islamic Studies Stream': 3, 'Arts and Music Stream': 4, 'Sports Stream': 5, 'Science stream but taking accounts instead of biology': 6})
data['Satisfaction'] = data['Satisfaction'].map({'Yes': 0, 'No': 1})

# Handle NaN values by dropping rows with NaN values
data.dropna(inplace=True)

# Split data into features and target
X = data.drop('Current Class Stream', axis=1)
y = data['Current Class Stream']

# Train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X, y)

# Load new data for prediction
data_2 = pd.read_csv('feed data.csv')

# Apply the same preprocessing steps as you did for the training data_2
data_2['CGPA'] = data_2['CGPA'].map({'Below 2.00': 0, '2.00 to 2.50': 1, '2.51 to 3.00': 2, '3.01 to 3.50': 3, '3.50 to 3.99': 4, '4': 5})
data_2['Class Performance [Bahasa Melayu]'] = data_2['Class Performance [Bahasa Melayu]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data_2['Class Performance [English]'] = data_2['Class Performance [English]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data_2['Class Performance [Sejarah]'] = data_2['Class Performance [Sejarah]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data_2['Class Performance [Mathematics]'] = data_2['Class Performance [Mathematics]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})
data_2['Class Performance [Science]'] = data_2['Class Performance [Science]'].map({'Bad': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5})

interests_list = ['Mathematics', 'Science (Physics, Chemistry, Biology)', 'Literature and Language Arts', 'History and Social Studies', 'Technology and Computer Science', 'Art and Creative Expression', 'Physical Education and Sports', 'Music and Performing Arts', 'Environmental Studies', 'Career and Vocational Interests']

for interest in interests_list:
    data_2['Interest_' + interest] = data_2['Interests'].apply(lambda x: 1 if interest in x else 0)
data_2.drop('Interests', axis=1, inplace=True)

data_2['Ambition'] = data_2['Ambition'].map({'Doctor or Healthcare Professional': 0, 'Engineer (e.g., Mechanical, Civil, Electrical)': 1, 'Computer Scientist or Software Developer': 2, 'Teacher or Educator': 3, 'Artist or Creative Professional': 4, 'Entrepreneur or Business Owner': 5, 'Musician or Performer': 6, 'Lawyer or Legal Professional': 7, 'Athlete or Sports Coach': 8, 'Scientist or Researcher': 8})

data_2['Current Class Stream'] = data_2['Current Class Stream'].map({'Science Stream': 0, 'Arts Stream': 1, 'Commerce Stream': 2, 'Islamic Studies Stream': 3, 'Arts and Music Stream': 4, 'Sports Stream': 5, 'Science stream but taking accounts instead of biology': 6})
data_2 = data_2.drop('Current Class Stream', axis=1)

data_2['Satisfaction'] = data_2['Satisfaction'].map({'Yes': 0, 'No': 1})

# Predict class probabilities
class_probabilities = classifier.predict_proba(data_2)[0]

# Visualize the probabilities
class_streams = ['Science Stream', 'Arts Stream', 'Commerce Stream', 'Islamic Studies Stream', 'Arts and Music Stream', 'Sports Stream']
plt.figure(figsize=(12,6))
plt.bar(class_streams, class_probabilities)
plt.title('Predicted Probabilities for Each Class Stream')
plt.ylabel('Probability')
plt.xlabel('Class Stream')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure to a .png file
plt.savefig('predicted_probabilities.png', dpi=300, bbox_inches='tight')

# Display the figure
plt.show()
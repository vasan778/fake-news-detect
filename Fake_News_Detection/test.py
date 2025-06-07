import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy.sparse import csr_matrix  # Updated import for SciPy compatibility

# Load dataset
data = pd.read_csv('train.csv')  # Update path
X = data['statement']
y = data['label']

# Convert labels if necessary
if y.dtype == 'object':
    y = y.map({'real': 0, 'fake': 1})  # Adjust based on your dataset

# Create and train the model
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # Adjust parameters as needed
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
pipeline = make_pipeline(vectorizer, model)

# Train the model
pipeline.fit(X, y)

# Save the model
with open('final_model.sav', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model retrained and saved as 'final_model.sav'")

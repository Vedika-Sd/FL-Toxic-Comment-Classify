import torch
from models import ToxicCommentModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load saved vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load test data 
df1 = pd.read_csv("Jigsaw_Toxic_Comment.csv")
df1 = df1[['comment_text', 'toxic']].dropna()
df1.columns = ['text', 'label']

# Vectorize
X = vectorizer.transform(df1['text']).toarray()
y = df1['label'].values

# Train-Test Split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to Tensor
X_test = torch.tensor(X_test, dtype=torch.float32)

# Load global model
model = ToxicCommentModel()
model.load_state_dict(torch.load('global_model.pth'))
model.eval()

# Predictions
with torch.no_grad():
    outputs = model(X_test)
    predictions = (outputs.numpy() > 0.5).astype(int).flatten()

# Accuracy and F1-Score
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"✅ Global Model Accuracy: {accuracy*100:.2f}%")

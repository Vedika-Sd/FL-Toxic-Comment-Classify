import torch
from models import ToxicCommentModel
import pickle

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load global model
model = ToxicCommentModel()
model.load_state_dict(torch.load('global_model.pth'))
model.eval()

# New text input
new_comments = ["You are disgusting!", "Nice video...", "You idiot fool", "Nice try dude", "I loved this idea", "Keep it up my boy", "I will go in court","Bad video, I hate it!"]

# Vectorize
X_new = vectorizer.transform(new_comments).toarray()
X_new = torch.tensor(X_new, dtype=torch.float32)

# Predict
with torch.no_grad():
    outputs = model(X_new)
    predictions = (outputs.numpy() > 0.5).astype(int).flatten()

# Print results
for comment, pred in zip(new_comments, predictions):
    label = "Toxic" if pred == 1 else "Not Toxic"
    print(f"{comment} --> {label}")

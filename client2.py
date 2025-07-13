# client2.py
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from models import ToxicCommentModel
import pickle

# Load vectorizer saved by Client 1
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load data
df1 = pd.read_csv("Jigsaw_Toxic_Comment.csv")
df1 = df1[['comment_text', 'toxic']].dropna()
df1.columns = ['text', 'label']

# Vectorize using same vectorizer
X = vectorizer.transform(df1['text']).toarray()
y = df1['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

# Model
model = ToxicCommentModel()

# Loss & Optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(7):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save local model
torch.save(model.state_dict(), 'client2_model.pth')

print("✅ Client 2 local model trained & saved.")

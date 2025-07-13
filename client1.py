# client1.py
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from models import ToxicCommentModel
import pickle

# Load data
df = pd.read_csv("toxicCR.csv")
df = df[['message', 'is_toxic']].dropna()
df.columns = ['text', 'label']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=300)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

# Save vectorizer for future clients
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

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
torch.save(model.state_dict(), 'client1_model.pth')

print("✅ Client 1 local model trained & saved.")

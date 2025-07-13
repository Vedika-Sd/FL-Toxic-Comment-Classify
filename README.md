# 📊 Federated Learning: Toxic Comment Detection (PyTorch)

This project demonstrates a simple Federated Learning (FL) workflow using PyTorch, where two clients train locally on their own data, and a server aggregates the models using Federated Averaging.

---

## 🚀 How to Run

### 📥 Install required libraries
```bash
pip install torch pandas scikit-learn
```
## 🚀 Execution Order

1️⃣ `models.py`   | Defines the neural network model  
2️⃣ `client1.py`  | Train model locally on Client 1 data  
3️⃣ `client2.py`  | Train model locally on Client 2 data  
4️⃣ `server.py`   | Aggregate client models using FedAvg  
5️⃣ `accuracy.py` | Test the final global model accuracy



# server.py
import torch
from models import ToxicCommentModel

# Load client models
client1_params = torch.load('client1_model.pth')
client2_params = torch.load('client2_model.pth')

# Initialize global model
global_model = ToxicCommentModel()

global_dict = {}
# FedAvg aggregation
n1 = 19651
n2 = 159571
total = n1 + n2

for key in client1_params.keys():
    global_dict[key] = (n1 * client1_params[key] + n2 * client2_params[key]) / total

# Load averaged weights
global_model.load_state_dict(global_dict)

# Save global model
torch.save(global_model.state_dict(), 'global_model.pth')

print("✅ Global model aggregated and saved as 'global_model.pth'")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cnn_model import CNNModel

# tensorize to 64, 64 and normalize to center tensor values
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder('archive/Training', transform=transform)
test_data = datasets.ImageFolder('archive/Testing', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# back prop and weight update
loss_values = []
num_of_epochs = 10
for epoch in range(num_of_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # avg loss of epoch
    epoch_loss = running_loss / len(train_loader)
    loss_values.append(epoch_loss)
    print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')

# evaluate against test data
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
print(f'Final Accuracy: {final_accuracy:.2f}%')

# save model, configurations, and states
torch.save({
    'epoch': num_of_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_values[-1],
}, 'cnn_brain_tumor_classification_v1.pth')

print("model saved")

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_of_epochs + 1), loss_values, marker='o', linestyle='-', linewidth=2)
plt.title('training loss over epochs')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=300)
plt.show()
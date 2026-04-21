# **FMNIST(Building a ANN)**

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# set manual seed for reproducibility(it meanns If you run the same code again,
# you should get the same results.)
torch.manual_seed(42)

df= pd.read_csv("/content/drive/MyDrive/fmnist_small.csv")
df

# Create a 4x4 grid for the image
fig,axes=plt.subplots(4,4,figsize=(10,10))
fig.suptitle("Fashion MNIST Dataset",fontsize=16)
# Plot the first 16 image from dataset
for i,ax in enumerate(axes.flatten()):
  img=df.iloc[i,1:].values.reshape(28,28)
  ax.imshow(img)
  ax.axis("off")
  ax.set_title(f"Label: {df.iloc[i,0]}")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()


# train test split
x=df.iloc[:,1:].values
y=df.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scaling
x_train=x_train/255.0
x_test=x_test/255.0


#create custom datset class
class CustomDataset(Dataset):
  def __init__(self, features,labels):
    self.labels=torch.tensor(labels,dtype=torch.long)
    self.features=torch.tensor(features,dtype=torch.float32)

  def __len__(self):
    return self.features.shape[0]
  def __getitem__(self,index):
    return self.features[index], self.labels[index]

#Creating train_dataset and test_dataset objects
train_dataset=CustomDataset(x_train,y_train)
test_dataset=CustomDataset(x_test,y_test)

len(train_dataset)

#creatings datloaders
train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=False)

#Defining NN class
class MyNN(nn.Module):
  def __init__(self,num_features):
    super().__init__()
    self.model=nn.Sequential(
        nn.Linear(num_features,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
  def forward(self,x):
    return self.model(x)

# set learning rate and epochs
epochs = 100
learing_rate=0.01

# instantiate the model
model=MyNN(x_train.shape[1])

#loss function
loss_function=nn.CrossEntropyLoss()

#optimizer
optimizer=optim.SGD(model.parameters(),lr=learing_rate)



len(test_dataloader)


loss_history = []

for epoch in range(epochs):
    total_epoch_loss = 0

    for batch_features, batch_labels in train_dataloader:

        outputs = model(batch_features)
        loss = loss_function(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss / len(train_dataloader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")



#set model to evaluation mode
model.eval()


total = 0
correct = 0

model.eval()

with torch.no_grad():
    for batch_features, batch_labels in test_dataloader:

        batch_features = batch_features.float()  # FIX HERE

        outputs = model(batch_features)

        _, predicted = torch.max(outputs, 1)

        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = correct / total
print("Test Accuracy:", accuracy * 100, "%")


total = 0
correct = 0

model.eval()

with torch.no_grad():
    for batch_features, batch_labels in train_dataloader:

        batch_features = batch_features.float()  # FIX HERE

        outputs = model(batch_features)

        _, predicted = torch.max(outputs, 1)

        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = correct / total
print("Train Accuracy:", accuracy * 100, "%")


import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Decreasing Over Epochs")
plt.show()


torch.save(model.state_dict(), "/content/drive/MyDrive/saved models/Fmnist.pth")
print("Model weights saved successfully!")


row_index = 5997

print("Label:", df.iloc[row_index, 0])
print("Pixel values:", df.iloc[row_index, 1:].values)


import matplotlib.pyplot as plt

row_index = 5997

label = df.iloc[row_index, 0]
image = df.iloc[row_index, 1:].values.reshape(28, 28)

plt.imshow(image, cmap="gray")
plt.title(f"Row {row_index} Label: {label}")
plt.axis("off")
plt.show()

# 👗 Fashion MNIST — ANN Classifier with PyTorch

A fully connected Artificial Neural Network (ANN) built from scratch using **PyTorch** to classify clothing items from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

---

## 📌 Project Overview

This project demonstrates end-to-end deep learning workflow:
- Loading and visualizing image data
- Custom PyTorch `Dataset` and `DataLoader` pipeline
- Building a multi-layer ANN
- Training with SGD optimizer and CrossEntropy loss
- Evaluating accuracy on train and test sets
- Saving and reloading model weights

---

## 🗂️ Project Structure

```
fashion-mnist-ann/
│
├── FMNIST_code.py          # Main training script
├── fmnist_small.csv       # Dataset (Fashion MNIST subset)
├── saved_models/
│   └── Fmnist.pth         # Saved model weights
├── Requirements.txt       # Python dependencies
└── README.md
```
## 4x4 grid for the image(Fmnist dataset in image)
<img width="629" height="520" alt="Screenshot 2026-04-21 200123" src="https://github.com/user-attachments/assets/2d2257d4-0336-496a-867b-164726d93b60" />




---

## 🧠 Model Architecture

```
Input Layer      →  784 neurons  (28×28 flattened pixels)
Hidden Layer 1   →  128 neurons  + ReLU activation
Hidden Layer 2   →  64 neurons   + ReLU activation
Output Layer     →  10 neurons   (one per class)
```

| Parameter       | Value             |
|----------------|-------------------|
| Optimizer       | SGD               |
| Learning Rate   | 0.01              |
| Loss Function   | CrossEntropyLoss  |
| Epochs          | 100               |
| Batch Size      | 32                |
| Train/Test Split| 80% / 20%         |

---

## 🏷️ Class Labels

| Label | Class       |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

## 📊 Results

| Split | Accuracy |
|-------|----------|
| Train | ~85–88%  |
| Test  | ~82–85%  |

> Loss decreases consistently over 100 epochs, confirming the model learns effectively.


## 📈 Training Loss Curve

The model trains for 100 epochs with loss plotted at the end to visualize convergence:

```python
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Decreasing Over Epochs")
plt.show()
```

---

## 💾 Saving & Loading the Model

**Save:**
```python
torch.save(model.state_dict(), "saved_models/Fmnist.pth")
```

**Load:**
```python
model = MyNN(784)
model.load_state_dict(torch.load("saved_models/Fmnist.pth"))
model.eval()
```

---

## 🛠️ Tech Stack

Python,
PyTorch,
Pandas,
scikit-learn,

## Training Loss decreasing over Epochs
<img width="570" height="487" alt="image" src="https://github.com/user-attachments/assets/900384ab-e635-4059-bc84-787c8de57638" />


---

## 🙋 Author

Ujala Soni


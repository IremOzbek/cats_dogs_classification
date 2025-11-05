# utils/visualize.py

import matplotlib.pyplot as plt
import os

def plot_training_history(history, output_dir="outputs/results"):
    os.makedirs(output_dir, exist_ok=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, 'bo-', label='Eğitim Doğruluğu')
    plt.plot(epochs, val_acc, 'ro-', label='Doğrulama Doğruluğu')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    acc_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(acc_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, 'bo-', label='Eğitim Kaybı')
    plt.plot(epochs, val_loss, 'ro-', label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kayıpları')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_path)
    plt.close()

    print(f" Eğitim grafikleri kaydedildi:")
    print(f"   - {acc_path}")
    print(f"   - {loss_path}")

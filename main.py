from utils.data_loader import load_data
from models.cnn_model import create_cnn_model
from utils.visualize import plot_training_history
import os


train_dir = "./data/train"
test_dir = "./data/test"


train_gen, test_gen = load_data(train_dir, test_dir)


model = create_cnn_model(input_shape=(150, 150, 3))


history = model.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen,
    verbose=1
)

plot_training_history(history, output_dir="outputs")

os.makedirs("outputs", exist_ok=True)
model.save("outputs/cat_dog_model.h5")

print(" Model başarıyla kaydedildi: outputs/cat_dog_model.h5")

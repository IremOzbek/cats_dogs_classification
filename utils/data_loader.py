import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(
        train_dir, test_dir,
        image_size=(150, 150),
batch_size=32,):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0/255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )
    print("\n ! Veri başarıyla yüklendi ve ön işlendi!")
    print(f" - Eğitim örnek sayısı: {train_generator.samples}")
    print(f" - Test örnek sayısı: {test_generator.samples}")

    return train_generator, test_generator
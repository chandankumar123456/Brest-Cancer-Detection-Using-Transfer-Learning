import os
import tensorflow as tf

def create_dummy_model(name="EfficientNetB0"):
    # Input shape: (None, 224, 224, 3)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # A couple of Conv2D layers so Grad-CAM has something to use
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    
    # Flatten and dense for classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

if __name__ == "__main__":
    saved_model_dir = "/Users/srivardhan/PULSE PROJECTS/March/Breast Cancer Detection/backend/saved_model"
    os.makedirs(saved_model_dir, exist_ok=True)
    
    # Create B0
    model_b0 = create_dummy_model("EfficientNetB0")
    model_b0.save(os.path.join(saved_model_dir, "model_B0.keras"))
    print("Saved dummy model_B0.keras")

    # Create B7 (for Grad-CAM)
    model_b7 = create_dummy_model("EfficientNetB7")
    model_b7.save(os.path.join(saved_model_dir, "model_B7.keras"))
    print("Saved dummy model_B7.keras")

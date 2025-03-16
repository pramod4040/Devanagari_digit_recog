import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

class EnhancedMoeModel:
    def __init__(self, weight_path=None, class_names=None):
        self.weight_path = weight_path
        self.model = None
        self.class_names = class_names
        self.input_shape = (32, 32, 3)
        self.num_classes = 46 if class_names is None else len(class_names)
        self.num_experts = 3
        
        # Create the model architecture
        self.create_model(
            input_shape=self.input_shape, 
            num_classes=self.num_classes, 
            num_experts=self.num_experts
        )
        
        # Load weights if path is provided
        if self.weight_path and os.path.exists(self.weight_path):
            self.load_weights()
    
    def create_efficientnet_b0_expert(self, input_shape, num_classes, name_prefix="expert_b0"):
        input_tensor = Input(shape=input_shape)
        resized_input = layers.Resizing(224, 224)(input_tensor)
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=resized_input
        )
        
        # Fine-tune the last few layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output_tensor = Dense(num_classes, activation='softmax', name=f'{name_prefix}_output')(x)
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')

    def create_efficientnet_b3_expert(self, input_shape, num_classes, name_prefix="expert_b3"):
        input_tensor = Input(shape=input_shape)
        resized_input = layers.Resizing(300, 300)(input_tensor)
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_tensor=resized_input
        )
        
        # Fine-tune the last few layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        output_tensor = Dense(num_classes, activation='softmax', name=f'{name_prefix}_output')(x)
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')

    def create_custom_cnn_expert(self, input_shape, num_classes, name_prefix="expert_cnn"):
        input_tensor = Input(shape=input_shape)
        
        # First convolutional block
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolutional block
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolutional block
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output_tensor = layers.Dense(num_classes, activation='softmax', name=f'{name_prefix}_output')(x)
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')

    def create_enhanced_gating_network(self, input_shape, num_experts):
        input_tensor = Input(shape=input_shape)
        
        # Convolutional layers for feature extraction
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output_tensor = layers.Dense(num_experts, activation='softmax')(x)
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name="enhanced_gating")

    def create_model(self, input_shape, num_classes, num_experts):
        input_layer = Input(shape=input_shape)
        
        # Create diverse experts
        expert1 = self.create_efficientnet_b0_expert(input_shape, num_classes, name_prefix="expert1")
        expert2 = self.create_efficientnet_b3_expert(input_shape, num_classes, name_prefix="expert2")
        expert3 = self.create_custom_cnn_expert(input_shape, num_classes, name_prefix="expert3")
        experts = [expert1, expert2, expert3]
        
        # Enhanced gating network
        gating_network = self.create_enhanced_gating_network(input_shape, num_experts)
        gate_outputs = gating_network(input_layer)
        expert_outputs = [expert(input_layer) for expert in experts]
        
        # Weighted combination of expert outputs
        moe_output = layers.Lambda(lambda inputs: tf.reduce_sum(
            tf.reshape(inputs[0], [-1, 1, num_experts]) * tf.stack(inputs[1], axis=2),
            axis=2
        ))([gate_outputs, expert_outputs])
        
        self.model = models.Model(inputs=input_layer, outputs=moe_output, name="enhanced_moe_model")

    def load_weights(self):
        """Load model weights from the specified path"""
        try:
            # Load weights by name, ignoring missing layers or shape mismatch
            self.model.load_weights(self.weight_path, by_name=True, skip_mismatch=True)
            print(f"Successfully loaded weights from {self.weight_path}")
            
            # Compile the model after loading weights
            optimizer = Adam(learning_rate=0.001)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
                        
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def save_weights(self, filepath):
        """Save model weights to the specified path"""
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")

    def predict(self, image_path):
        """
        Make predictions on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with predicted class and confidence score
        """
        # Load and preprocess the image
        img = Image.open(image_path).resize((32, 32)).convert("RGB")
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        
        # Make prediction
        predictions = self.model.predict(img_array)
        scores = tf.nn.softmax(predictions)
        
        # Get prediction results
        predicted_class_idx = np.argmax(scores)
        confidence = float(scores[0][predicted_class_idx])
        
        if self.class_names and len(self.class_names) > predicted_class_idx:
            predicted_label = self.class_names[predicted_class_idx]
        else:
            predicted_label = f"Class {predicted_class_idx}"
        
        return {
            'class': predicted_label,
            'confidence': confidence
        }
    
    def predict_batch(self, images):
        """
        Make predictions on a batch of images
        
        Args:
            images: Batch of images as numpy array [batch_size, height, width, channels]
            
        Returns:
            Array of dictionaries with predicted classes and confidence scores
        """
        # Ensure images are normalized
        if images.max() > 1.0:
            images = images / 255.0
            
        predictions = self.model.predict(images)
        results = []
        
        for i, pred in enumerate(predictions):
            scores = tf.nn.softmax(pred)
            predicted_class_idx = np.argmax(scores)
            confidence = float(scores[predicted_class_idx])
            
            if self.class_names and len(self.class_names) > predicted_class_idx:
                predicted_label = self.class_names[predicted_class_idx]
            else:
                predicted_label = f"Class {predicted_class_idx}"
            
            results.append({
                'class': predicted_label,
                'confidence': confidence
            })
        
        return results
    
    def compile(self, learning_rate=0.001):
        """Compile the model with specified parameters"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model compiled successfully.")

    def summary(self):
        """Display model summary"""
        return self.model.summary()


# Helper function to get class names
def get_class_names():
    class_names = [
        'character_10_yna',
        'character_11_taamatar',
        'character_12_thaa',
        'character_13_daa',
        'character_14_dhaa',
        'character_15_adna',
        'character_16_tabala',
        'character_17_tha',
        'character_18_da',
        'character_19_dha',
        'character_1_ka',
        'character_20_na',
        'character_21_pa',
        'character_22_pha',
        'character_23_ba',
        'character_24_bha',
        'character_25_ma',
        'character_26_yaw',
        'character_27_ra',
        'character_28_la',
        'character_29_waw',
        'character_2_kha',
        'character_30_motosaw',
        'character_31_petchiryakha',
        'character_32_patalosaw',
        'character_33_ha',
        'character_34_chhya',
        'character_35_tra',
        'character_36_gya',
        'character_3_ga',
        'character_4_gha',
        'character_5_kna',
        'character_6_cha',
        'character_7_chha',
        'character_8_ja',
        'character_9_jha',
        'digit_0',
        'digit_1',
        'digit_2',
        'digit_3',
        'digit_4',
        'digit_5',
        'digit_6',
        'digit_7',
        'digit_8',
        'digit_9'
    ]
    return class_names

# Example of how to use the model
if __name__ == "__main__":
    # Path to your saved model weights
    weights_path = "path/to/your/model/weights.h5"
    
    # Get class names
    class_names = get_class_names()
    
    # Create model instance and load weights
    model = EnhancedMoeModel(weight_path=weights_path, class_names=class_names)
    
    # Optional: Compile the model with custom learning rate
    # model.compile(learning_rate=0.0005)
    
    # Display model summary
    model.summary()
    
    # Make a prediction
    result = model.predict("path/to/test/image.jpg")
    print(f"Predicted class: {result['class']}, Confidence: {result['confidence']}")
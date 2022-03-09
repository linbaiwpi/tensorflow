import tensorflow as tf
import tensorflow.keras as K

# Create a model using high-level tf.keras.* APIs
input1 = tf.keras.layers.Input(shape=(2,))
input2 = tf.keras.layers.Input(shape=(2,))
added = tf.keras.layers.Add()([input1, input2])
model = tf.keras.models.Model(inputs=[input1, input2], outputs=added)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model_add.tflite', 'wb') as f:
  f.write(tflite_model)


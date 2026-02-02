import tensorflow as tf

model = tf.keras.models.load_model(
    "models/rdkit_maccs_cnn.keras",
    compile=False
)

model.summary()

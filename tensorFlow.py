import tensorflow as tf

# Define the input features
features = tf.keras.Input(shape=(n_features,), name="input_features")

# Add a dense layer with 64 units and ReLU activation
x = tf.keras.layers.Dense(64, activation="relu")(features)

# Add a dropout layer to prevent overfitting
x = tf.keras.layers.Dropout(0.5)(x)

# Add another dense layer with 32 units and ReLU activation
x = tf.keras.layers.Dense(32, activation="relu")(x)

# Add a dropout layer to prevent overfitting
x = tf.keras.layers.Dropout(0.5)(x)

# Add a final dense layer with a single unit and sigmoid activation to output a probability score
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# Define the model
model = tf.keras.Model(inputs=features, outputs=output)

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model on the prepared data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

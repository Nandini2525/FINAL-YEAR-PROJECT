from keras.layers import LSTM, Dense, Bidirectional
from sklearn.decomposition import PCA
# Assuming you have X and y loaded
X_arr = X.to_numpy()

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_arr)

# Apply PCA for feature reduction
pca = PCA(n_components=30)  # Adjust the number of components as needed
X_pca = pca.fit_transform(X_scaled)

# Reshape data for LSTM
X_lstm = X_pca.reshape(X_pca.shape[0], 1, X_pca.shape[1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Define the Bidirectional LSTM model
lstm_model = Sequential()
lstm_model.add(Bidirectional(LSTM(10), input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dense(1, activation='tanh'))

# Compile and train the model
lstm_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
a=lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
accuracy_bilstm_pca = lstm_model.evaluate(X_test, y_test)[1]
print("Test accuracy:", accuracy_bilstm_pca)

plt.plot(a.history['loss'])
plt.plot(a.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Train the Bidirectional LSTM model
history_bilstm = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Obtain predictions on the test set
y_pred_proba = lstm_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Plot metrics in a bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(4, 4))
plt.bar(metrics, values, color=['skyblue', 'green', 'orange', 'pink'])
plt.title('Model Metrics')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.show()

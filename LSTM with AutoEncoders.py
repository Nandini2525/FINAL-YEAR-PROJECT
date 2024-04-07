from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split

import numpy as np


X_arr = X.to_numpy()
# Flatten the time steps dimension
X_flat = X_arr.reshape(X_arr.shape[0], -1)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# Reshape back to 3D for LSTM
X_lstm = X_scaled.reshape(X.shape[0], X.shape[1], -1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Define the autoencoder model
autoencoder = Sequential()
autoencoder.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
autoencoder.add(Dense(X_train.shape[1], activation='linear'))

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(30, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dense(1, activation='sigmoid'))

# Combine both models
input_layer = Input(shape=(X_train.shape[1],))
autoencoder_out = autoencoder(input_layer)
lstm_out = lstm_model(autoencoder_out)

full_model = Model(inputs=input_layer, outputs=lstm_out)

# Compile and train the model
full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=full_model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the model on the test set
accuracy = full_model.evaluate(X_test, y_test)[1]
print('Accuracy:', accuracy)

y_pred_proba = full_model.predict(X_test)
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

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
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

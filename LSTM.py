from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


input_shape = (X_train.shape[1], 1)

# Build the LSTM model without regularization
 # Increased epochs
model = Sequential()
model.add(LSTM(30, input_shape=input_shape))  # Increased number of LSTM units
model.add(Dense(1, activation='tanh'))  # Output layer with sigmoid activation

model.summary()

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=72, verbose=2, shuffle=False)  # Increased epochs

# Evaluate the model on the test set
result = model.predict(X_test)
y_pred = [1 if res > 0.5 else 0 for res in result]

accuracy_lstm = model.evaluate(X_test,y_test)
print('Accuracy:', accuracy_lstm) 

y_pred_proba = model.predict(X_test)
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

import matplotlib.pyplot as plt

# Plot training & validation loss values
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

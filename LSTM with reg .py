from keras.layers import Dense, LSTM, Dropout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


input_shape = (X_train.shape[1], 1)

# Build the LSTM model with regularization

model = Sequential()
model.add(LSTM(50, input_shape=input_shape))  # Increased number of LSTM units
model.add(Dropout(0.3))  # Increased dropout rate to 0.5
model.add(Dense(1, activation='relu'))  # Output layer with sigmoid activation

model.summary()

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=72, verbose=2, shuffle=False)

# Evaluate the model on the test set
result = model.predict(X_test)
y_pred = [1 if res > 0.5 else 0 for res in result]

accuracy_lstm_regularization = model.evaluate(X_test,y_test)
print('Accuracy:', accuracy_lstm_regularization)

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

import matplotlib.pyplot as plt

# Define the datasets and models
datasets = ['SDN', 'CICIDS 2018']
models = ['LSTM', 'LSTM with regularization', 'LSTM with PCA', 'LSTM with MI', 'LSTM with auto encoders', 'BiLSTM with PCA']
accuracies = [[0.79, 0.83, 0.96, 0.88, 0.99, 0.96], [0.80, 0.83, 0.91, 0.92, 0.95, 0.97]]

# Create a bar graph
fig, ax = plt.subplots(figsize=(8, 6))

# Define bar width
bar_width = 0.35

# Define index for x-axis
index = range(len(models))

# Plot bars for each dataset
for i, dataset in enumerate(datasets):
    ax.bar([x + i * bar_width for x in index], accuracies[i], bar_width, label=dataset)

# Set labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of LSTM Models on SDN and CICIDS 2018 Datasets')
ax.set_xticks([x + bar_width for x in index])
ax.set_xticklabels(models)
ax.legend()

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

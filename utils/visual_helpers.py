import matplotlib.pyplot as plt

# plot training loss over epochs
def plot_train_loss(trainer, num_epochs=100):
  # load values
  epochs = list(range(1, num_epochs + 1))
  losses = trainer.get_loss()

  # plot graph
  plt.figure(figsize=(8, 5))
  plt.scatter(epochs, losses, color='blue', label='Training Loss per Epoch')
  plt.plot(epochs, losses, color='lightblue', linestyle='--')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training Loss over Epochs')
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
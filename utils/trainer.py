# general class for training models
class Trainer:
    # intialize trainer
    def __init__(self, model, optimizer, train_loader, device, epochs=10):
        self.model = model                # the model to train
        self.optimizer = optimizer        # optimizer chosen to train the model
        self.train_loader = train_loader  # data loader with the training data
        self.device = device              # device: CPU or GPU (cuda)
        self.epochs = epochs              # number of epochs
        self.tracked_loss = []            # average loss tracked over epochs

    # forward pass will be specific for each model
    def fwd_pass(self, input_data):
        raise NotImplementedError

    # training the model
    def train(self):
        # reset the tracked losses
        self.tracked_loss = []

        # training loop
        for epoch in range(self.epochs):
            print(f"\nEPOCH {epoch + 1} -----------------------------")

            # loss tracking variables
            total_loss = 0.0
            num_batches = 0

            # iterate over batches
            for input_data in self.train_loader:
                # forward pass
                loss = self.fwd_pass(input_data)

                # backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # track loss
                total_loss += loss.item()
                num_batches += 1

            # track loss over epochs
            avg_loss = total_loss / num_batches
            self.tracked_loss.append(avg_loss)

            # print loss
            print(f"[average loss: {avg_loss:.4f}]")

    # returns the list of average losses
    def get_loss(self):
        return self.tracked_loss
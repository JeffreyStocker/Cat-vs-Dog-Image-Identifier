from model_save import save_universal_model
from model_validate import validate_model

def train(model, criterion, optimizer, training_data, test_data, save_data, epochs = 1, should_save_checkpoint = True, device='cpu'):

    def save_function(current_epoch):
        if save_data:
            return save_universal_model(model, save_data, epochs=current_epoch)
        else:
            return lambda x: x

    initial_epoch = 0

    if save_data and save_data.get('epochs'):
        epochs += save_data.get('epochs')
        initial_epoch = save_data.get('epochs')

    model.to(device)
    model.train()

    train_losses, test_losses = [], []

    if initial_epoch > 0:
        print(f'Resuming Training Starting at epoch: {str(initial_epoch + 1)}, Running for {epochs - initial_epoch} epochs')

    print('***** Start Training *****')
    for epoch in range(initial_epoch, epochs):
        running_loss = 0
        steps = 0
        total_steps = len(training_data)
        for images, labels in training_data:
            steps += 1
            print (f'Epoch: {epoch + 1}/{epochs}: Step {steps}/{total_steps} IN PROGRESS...', end="\r")

            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            log_percent = model.forward(images)
            loss = criterion(log_percent, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_loss, accuracy, test_losses = validate_model(model, test_data, criterion, test_losses, device=device)

        train_losses.append(running_loss/len(training_data))
        test_losses.append(test_loss/len(test_data))

        if should_save_checkpoint:
            save_function(epoch + 1)

        print('----------------------------------------')
        print(f'Epoch: {epoch + 1}/{epochs} | '
                f'Training Loss: {train_losses[-1]:.4f} | '
                f'Test Loss: {test_losses[-1]:.4f} | '
                f'Test Accuracy: {accuracy:.2f}% | '
                )

    print('***** Done Training *****')
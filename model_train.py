from model_save import save_universal_model
from model_validate import validate_model

def train(model, criterion, optimizer, training_data, test_data, epochs = 1, learning_rate = .003, should_save_checkpoint = True, save_data=None, device='cpu' ):

    def save_function(current_epoch, extra_text = '', include_epoch_in_title = False):
        if save_data:
            return save_universal_model(model, save_data, epochs = current_epoch, include_epoch_in_title=include_epoch_in_title)
        else:
            return lambda x: x

    initial_epoch = 0

    if save_data and save_data.get('epochs'):
        epochs += save_data.get('epochs')
        initial_epoch = save_data.get('epochs')

    model.to(device)
    model.train()

    criterion = criterion()
    optimizer = optimizer(model.classifier.parameters(), learning_rate)

    steps = 0
    steps_to_evaluate = 10

    train_losses, test_losses = [], []

    print('***** Start Training *****')
    for epoch in range(initial_epoch, epochs):
        running_loss = 0

        for images, labels in training_data:
            steps += 1
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            log_percent = model.forward(images)
            loss = criterion(log_percent, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            break
            if steps % steps_to_evaluate == 0:
                pass

        test_loss, accuracy, test_losses = validate_model(model, test_data, criterion, test_losses, save_function, device=device)

        train_losses.append(running_loss/len(training_data))
        test_losses.append(test_loss/len(test_data))

        if should_save_checkpoint:
            save_function(epoch, extra_text="Mid")

        print('----------------------------------------')
        print(f'E.: {epoch + 1}/{epochs} | '
                f'Training Loss: {train_losses[-1]:.4f} | '
                f'Test Loss: {test_losses[-1]:.4f} | '
                f'Test Accuracy: {accuracy:.2f} | '
                )
    if should_save_checkpoint:
        save_function(epoch, include_epoch_in_title=True)

    print('***** Done Training *****')
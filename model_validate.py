import torch

def validate_model(model, test_data, criterion, losses_ = None, device='cpu'):
    test_loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)

            loss = criterion(logps, labels)
            test_loss += loss.item()

            percent = torch.exp(logps)

            top_p, top_class = percent.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    model.train()

    accuracy = accuracy/len(test_data)* 100

    if losses_:
        losses_.append(test_loss/len(test_data))

    return test_loss, accuracy, losses_
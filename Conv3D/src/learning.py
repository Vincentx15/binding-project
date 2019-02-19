import time
import torch
import sys


def test(model, test_loader, test_loss_fn, device):
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """
    model.eval()

    test_loss = 0
    test_size = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        output = model(inputs)
        test_size += len(inputs)
        test_loss += test_loss_fn(output, targets)
    test_loss /= test_size
    return test_loss


def train_model(model, criterion, optimizer, device, train_loader, validation_loader, save_path,
                writer=None, num_epochs=25, wall_time=.25):
    """
    Performs the entire training routine. Save the model in the save_path
        device:
        dataloaders:  and validation
        writer (Writer): the writer for gathering training data (loss and accuracy)
        num_epochs (int): the number of epochs for training
        log_interval (int): controls the rate of logs
        wall_time (float): the wall time (in hours) given to the compute node, to make sure everything is saved

    Returns:
        model: the best model among all epochs.

    :param model: (torch.nn.Module): the model to train
    :param criterion: the criterion to use (eg CrossEntropy)
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param device: the device on which to run
    :param train_loader: dataloader for training
    :param validation_loader: dataloader for validation
    :param save_path: where to save the model
    :param writer: a Tensorboard object (defined in utils)
    :param num_epochs: int number of epochs
    :param wall_time: The number of hours you want the model to run
    :return:
    """

    epochs_from_best = 0
    early_stop_threshold = 10

    start_time = time.time()
    best_loss = sys.maxsize

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Training phase
        model.train()

        running_loss = 0.0

        num_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_size = len(inputs)

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # running_corrects += labels.eq(target.view_as(out)).sum().item()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * batch_size,
                    num_batches * batch_size,
                    100. * batch_idx / num_batches,
                    loss.item()))

                # tensorboard logging
                writer.log_scalar("Training loss", loss.item(),
                                  epoch * num_batches + batch_idx)

        # Log training metrics

        train_loss = running_loss / num_batches
        writer.log_scalar("Train loss during training", train_loss, epoch)

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss = test(model, validation_loader, criterion, device)
        writer.log_scalar("Test loss during training", test_loss, epoch)
        # writer.log_scalar("Test accuracy during training", test_accuracy, epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_from_best = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)

        # Early stopping
        else:
            epochs_from_best += 1
            if epochs_from_best > early_stop_threshold:
                print('This model was early stopped')
                break


        # Break out of the loop if we might go beyond the wall time
        time_elapsed = time.time() - start_time
        if time_elapsed * (1 + 1 / (epoch + 1)) > .95 * wall_time * 3600:
            break
    return best_loss


def make_predictions(data_loader, model, optimizer, model_weights_path):
    """
    :param data_loader: an iterator on input data
    :param model: An empty model
    :param optimizer: An empty optimizer
    :param model_weights_path: the path of the model to load
    :return: list of predictions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    predictions = []

    for batch_idx, inputs in enumerate(data_loader):
        inputs = inputs.to(device)
        predictions.append(model(inputs))
    return predictions


if __name__ == "__main__":
    pass
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='../data/testset')
# parser.add_argument('--out_dir', default='Submissions/')
# parser.add_argument(
#     '--model_path', default='results/base_wr_lr01best_model.pth')
# args = parser.parse_args()
# make_predictions(args.data_dir, args.out_dir, args.model_path)

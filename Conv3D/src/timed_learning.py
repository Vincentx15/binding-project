import time
import torch
import sys
from src.utils import debug_memory


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
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        test_size += len(inputs)
        test_loss += test_loss_fn(output, targets).item()
        if not batch_idx % 50 :
            print('Computed {:.2f} batches for the test'.format(batch_idx))
    test_loss /= test_size
    return test_loss


def train_model(model, criterion, optimizer, device, train_loader, validation_loader, save_path,
                writer=None, num_epochs=25, wall_time=None):
    """
    Performs the entire training routine.
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

        end_g = 0
        num_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_size = len(inputs)

            start_loop = time.perf_counter()
            inputs, labels = inputs.to(device), labels.to(device)
            torch.cuda.synchronize()  # wait for mm to finish
            b = time.perf_counter() - start_loop
            print('send_data {:.02e}s'.format(b))

            a = time.perf_counter()
            out = model(inputs)
            torch.cuda.synchronize()  # wait for mm to finish
            c = time.perf_counter() - a
            print('forward {:.02e}s'.format(c))

            loss = criterion(out, labels)

            a = time.perf_counter()
            loss.backward()

            torch.cuda.synchronize()  # wait for mm to finish
            e = time.perf_counter() - a
            print('Backward {:.02e}s'.format(e))

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            # running_corrects += labels.eq(target.view_as(out)).sum().item()
            if batch_idx % 20 == 0:
                time_elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Time: {:.2f}'.format(
                    epoch,
                    batch_idx * batch_size,
                    num_batches * batch_size,
                    100. * batch_idx / num_batches,
                    loss.item(),
                    time_elapsed))

                # tensorboard logging
                writer.log_scalar("Training loss", loss.item(),
                                  epoch * num_batches + batch_idx)

            g = time.perf_counter()
            print('Loop total {:.02e}s'.format(g - start_loop))
            print('Loop total with for {:.02e}s'.format(g - end_g))
            end_g = time.perf_counter()
            print(torch.cuda.memory_allocated(device=device))
            print(torch.cuda.memory_cached(device=device))

        # Log training metrics
        train_loss = running_loss / num_batches
        writer.log_scalar("Train loss during training", train_loss, epoch)

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss = test(model, validation_loader, criterion, device)
        writer.log_scalar("Test loss during training", test_loss, epoch)
        # writer.log_scalar("Test accuracy during training", test_accuracy, epoch)

        # Checkpointing
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

        # Sanity Check
        if wall_time is not None:
            # Break out of the loop if we might go beyond the wall time
            time_elapsed = time.time() - start_time
            if time_elapsed * (1 + 1 / (epoch + 1)) > .95 * wall_time * 3600:
                break
    return best_loss


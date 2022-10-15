from tqdm import tqdm
import torch
# training

def train1Epoch(epoch_index, model, optimizer, loss_fn, training_loader, tb_writer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(training_loader), total=len(training_loader)):
        # Every data instance is an input + label pair
        image, bnpp = data
        image, bnpp = image.to(device), bnpp.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(image)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(), bnpp)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
import torch

def train_step(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
               device:torch.device, print_step:int=100):

    running_loss = 0.0
    total_loss = 0.0
    counter = 0.0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        counter += 1.
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
    return total_loss / counter
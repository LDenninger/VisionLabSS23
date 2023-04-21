import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pathlib import Path as P

from tqdm import tqdm

from experiments import Logger


def train_model(model: nn.Module, 
                train_dataset: torch.utils.data.DataLoader, 
                eval_dataset: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                config: dict, 
                logger: Logger = None,
                verbose: bool = False):
    """
    Train the model on the provided dataset.

    Parameters:
        model (nn.Module): Model to train.
        train_dataset (torch.utils.data.DataLoader): Dataset to train the model on.
        eval_dataset (torch.utils.data.DataLoader): Dataset to evaluate the model on.
        config (dict): Dictionary containing the configuration of the model.
    """
    import ipdb; ipdb.set_trace()


    ###----Hyperparameters----###
    RUN_NAME = config["run_name"]
    EPOCHS = config["num_epochs"]

    EVAL_FREQUENCY = config["eval_frequency"]
 
    VERBOSE = verbose

    ###----Training----###

    # Display
    output = f"----Training for model {RUN_NAME}----\n\nHyperparameters:\n "
    for param in config:
        output += f"  {param}: {config[param]}\n"
    output += "Start training...\n"
    print(output)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Train the model
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}----------')

        # Train for one epoch
        losses = run_epoch(model=model, dataset=train_dataset, criterion=criterion, optimizer=optimizer, device=device, verbose=VERBOSE)

        # Evaluate on validation set.
        if epoch % EVAL_FREQUENCY == 0:
            accuracy = run_epoch(model=model, dataset=eval_dataset, device=device, verbose=VERBOSE)

        if logger is not None:
            logger.step(
                epoch=epoch,
                data={
                    "loss": losses,
                    "accuracy": accuracy,
                }
            )



    
def run_epoch(model: nn.Module, dataset: torch.utils.data.DataLoader, criterion: nn.Module=None, optimizer: torch.optim.Optimizer=None, device: str="cpu", verbose: bool=False):
    """
    Run a single epoch of training or evaluation. Providing an optimizer and a criterion is required to train the model, else the model is evaluated on the provided dataset.
    
    """
    assert (criterion is not None and optimizer is not None) or (criterion is None and optimizer is None)


    if optimizer is not None:
        model.train()
        loss_list = []
        train=True
    else:
        model.eval()
        n_correct = 0
        train=False

    progress_bar = tqdm(dataset, total=len(dataset))
    import ipdb; ipdb.set_trace()

    for i, (imgs, labels) in enumerate(progress_bar):
        # Prepare inputs
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = imgs.flatten(start_dim=1)

        # reset all gradients
        if train==True:
            optimizer.zero_grad()

        # Produce output
        outputs = model(imgs)

        if train==True:
            # Compute loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()

            loss_list.append(loss.item())

            # Finally update all weights
            optimizer.step()

        else:
            pred_labels = torch.argmax(outputs, dim=-1)

            correct_labels = len(torch.where(pred_labels == labels)[0])
            n_correct += correct_labels


        progress_bar.set_description(f'Loss/: {loss.item():.4f}' if train else f'Accuracy: {(n_correct / i):.4f}')

    if verbose:
        # Additionally returning the loss list after training
        return loss_list if train else (n_correct / len(dataset))

    return True

    


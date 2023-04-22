import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pathlib import Path as P

from tqdm import tqdm

from experiments import Logger

### Training Scripts ###

def train_model(model: nn.Module, 
                train_dataset: torch.utils.data.DataLoader, 
                eval_dataset: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                config: dict, 
                logger: Logger = None,
                verbose: bool = True):
    """
    Train the model on the provided dataset.

    Parameters:
        model (nn.Module): Model to train.
        train_dataset (torch.utils.data.DataLoader): Dataset to train the model on.
        eval_dataset (torch.utils.data.DataLoader): Dataset to evaluate the model on.
        config (dict): Dictionary containing the configuration of the model.
    """


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
        accuracy = -1
        # Train for one epoch   
        losses = run_epoch(model=model, dataset=train_dataset, criterion=criterion, optimizer=optimizer, device=device, verbose=True)

        # Evaluate on validation set.
        if epoch % EVAL_FREQUENCY == 0:
            accuracy = run_epoch(model=model, dataset=eval_dataset, device=device, verbose=VERBOSE)
        if logger is not None:
            
            logger.step(
                epoch=epoch+1,
                data={
                    "loss": sum(losses)/len(losses),
                    "accuracy": accuracy ,
                },
                model=model,
                optimizer=optimizer
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


    for i, (imgs, labels) in enumerate(progress_bar):
        # Prepare inputs
        imgs = imgs.to(device)
        labels = labels.to(device)

        labels_ohc = _one_hot_encoding(labels, 10)

        imgs = _rgb2grayscale(imgs).squeeze()
        imgs = _flatten_img(imgs)


        # reset all gradients
        if train==True:
            optimizer.zero_grad()

        # Produce output
        outputs = model(imgs)

        if train==True:
            # Compute loss and backpropagate
            loss = criterion(outputs, labels_ohc)
            loss.backward()

            loss_list.append(loss.item())

            # Finally update all weights
            optimizer.step()

        else:
            pred_labels = torch.argmax(outputs, dim=-1)

            correct_labels = len(torch.where(pred_labels == labels)[0])
            n_correct += correct_labels


        progress_bar.set_description(f'Loss/: {loss.item():.4f}' if train else f'Accuracy: {(n_correct / (i+1)):.4f}')

    if verbose:
        # Additionally returning the loss list after training
        return loss_list if train else (n_correct / len(dataset))

    return True

    

### Data Pre-Processing ###

def _flatten_img(input: torch.Tensor, only_img_size: bool=True):
    # Flatten only the image size dimensions

    if only_img_size:
        if len(input.shape)==4:
            return torch.flatten(input, start_dim=2)
        if len(input.shape)==3:
            return torch.flatten(input, start_dim=1)
    # Flatten all dimensions except of the batch dimension
    else:
        return torch.flatten(input, start_dim=1)

def _rgb2grayscale(input: torch.Tensor):
    return tv.transforms.Grayscale()(input)

def _rgb2hsv(input: torch.Tensor):
    """
    Adapted from: https://github.com/limacv/RGB_HSV_HSL.git
    """
    cmax, cmax_idx = torch.max(input, dim=1, keepdim=True)
    cmin = torch.min(input, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(input[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((input[:, 1:2] - input[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((input[:, 2:3] - input[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((input[:, 0:1] - input[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(input), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

def _hsv2rgb(input):
    """
    Adapted from: https://github.com/limacv/RGB_HSV_HSL.git
    """
    hsv_h, hsv_s, hsv_l = input[:, 0:1], input[:, 1:2], input[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

def _one_hot_encoding(labels: torch.tensor, num_classes: int):
    return fun.one_hot(labels, num_classes=num_classes).float()
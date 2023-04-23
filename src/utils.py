import torch
import torch.nn.functional as fun
import torchvision as tv


####---- Loading/Initialization Scripts ----####
def initialize_optimizer(model: torch.nn.Module, config: dict):
    """
        Initialize optimizer.

        Arguments:
            model (torch.nn.Module): Model to be optimized.
            config (dict): Optimizer configuration dictionary.
                Format:
                        {
                            'type': 'Adam',
                            'learning_rate': Learning rate (float),
                            'betas': Betas (tuple of floats),
                            'eps': Epsilon (float),
                        }
    """
    if config['type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['betas']), eps=config['eps'])
        
    return optimizer

def initialize_loss( config: dict):
    """
        Initialize criterion.

        Arguments:
            config (dict): Criterion configuration dictionary.
                Format:
                        {
                            'type': 'CrossEntropyLoss',
                        }
    """
    if config['type'] == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()

    return criterion



####---- Data Preprocessing ----####

def apply_data_preprocessing(images: torch.Tensor, labels: torch.tensor, config: dict):
    """
    Apply data preprocessing to images and labels.

    Arguments:
        images (torch.Tensor): Images to be preprocessed.
        labels (torch.tensor): Labels to be preprocessed.
        config (dict): Pre-processing configuration dictionary.
            Format:
                    {
                        'flatten': Flatten image (bool),
                        'flatten_only_img_size': Only flatten the height and width dimensions (bool),
                        'rgb2gray': Convert RGB to grayscale (bool),
                        'lbl_oneHot': Convert labels to one-hot format (bool),
                        'squeeze': Squeeze singleton dimensions of the image (bool),
                    }
    """
    images_proc = images.clone()
    labels_proc = labels.clone()

    if config['rgb2gray']:
        images_proc = _rgb2grayscale(images_proc)

    if config['flatten']:
        images_proc = _flatten_img(images_proc, only_img_size=config['flatten_only_img_size'])
    
    if config['lbl_oneHot']:
        labels_proc = _one_hot_encoding(labels_proc)

    return images_proc, labels_proc
## Processing Functions ##    

def _flatten_img(input: torch.Tensor, only_img_size: bool=True):
    # Flatten only the image size dimensions

    if only_img_size:
        return torch.flatten(input, start_dim=-2)
   
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
    rgb = torch.empty_like(input)
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
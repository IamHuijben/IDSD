import numpy as np 
import torch
import copy
import subprocess
import sys
from pathlib import Path

def tensor2array(tensor):
    """Converts a pytorch tensor to a numpy array

    Args:
        tensor (torch.tensor): A tensor 

    Returns:
        np.ndarray: A numpy array
    """

    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return np.array(tensor)
    if tensor.is_cuda or tensor.is_mps:
        tensor = tensor.clone().to('cpu')
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.numpy()
    return tensor


def fix_random_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("Using CPU.")
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device

def get_git_commit_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip(), 'utf-8')

def get_git_branch():
    return str(subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip(), 'utf-8')

def make_unique_path(save_dir):
    if not save_dir.parent.exists():
        save_dir.parent.mkdir(exist_ok=False, parents=True)

    try:
        save_dir.mkdir(exist_ok=False, parents=True)
    except:
        unique_dir_found = False
        post_fix = 0
        while not unique_dir_found:
            try:
                Path(str(save_dir) +
                        f"_{post_fix}").mkdir(exist_ok=False, parents=True)
                unique_dir_found = True
                save_dir = Path(str(save_dir) + f"_{post_fix}")
            except:
                post_fix += 1
    return save_dir

def prepare_dict_for_yaml(my_dict):
    if not my_dict: return my_dict 
    
    # Convert all values in the dict to floats rather than numpy arrays or tensors for storage to yml file.
    for name, value in my_dict.items():
        if isinstance(value, dict):
            value = prepare_dict_for_yaml(value)
        else:
            if 'torch' in sys.modules and isinstance(value, torch.Tensor):
                value = tensor2array(value).tolist()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            
            if isinstance(value, list):
                if len(value)>0 and isinstance(value[0], str):
                    my_dict[name] = value
                else:
                    my_dict[name] = [float(val) for val in value]
            elif isinstance(value, str):
                my_dict[name] = value
            elif isinstance(value, Path):
                my_dict[name] = str(value)
            elif np.char.isdigit(str(value)):
                my_dict[name] = float(value)
            else:
                try:
                    my_dict[name] = float(value)
                except:
                    my_dict[name] = value

    if isinstance(list(my_dict.keys())[0],float):
        new_dict = copy.deepcopy(my_dict)
        for name,value in my_dict.items():
            new_dict[int(name)] = value
            new_dict.pop(name)

        my_dict = new_dict

    return my_dict
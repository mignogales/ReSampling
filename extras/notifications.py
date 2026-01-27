
from colorama import Fore, Style

def notify_update(key, wandb_keys, wandb_config, cfg, sub_attr=None):
    """
    Dynamically updates a nested config attribute and notifies the user.
    """
    if key in wandb_config:
        new_value = wandb_config[key]
        
        # Start at the root configuration
        target_obj = cfg
        
        # If there are sub-attributes (e.g., "model.params.hyper"), traverse them
        if sub_attr:
            for attr in sub_attr.split('.'):
                target_obj = getattr(target_obj, attr)
        
        # target_obj is now a reference to the specific nested object
        setattr(target_obj, key, new_value)
        
        # Dynamic notification
        path_display = f"{sub_attr}.{key}" if sub_attr else key
        print(f"{Fore.GREEN}Updated {path_display}: {new_value}{Style.RESET_ALL}")

    return cfg
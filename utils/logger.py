import os
import datetime
import numpy as np
import wandb

class LocalLogger:
    def __init__(self) -> None:
        pass

    def update(self, log_info):
        pass

    def log(self):
        pass


class WandbLogger:
    def __init__(self, opts) -> None:
        run_name = os.path.basename(opts["exp_dir"])
        wandb.init(project="ExpressionSR", entity="jwy20", config=opts, name=run_name)
    
    @staticmethod
    def log_best_model():
        wandb.run.summary["best-model-save-time"] = datetime.datetime.now()

    @staticmethod
    def log(prefix, metrics_dict, global_step):
        log_dict = {f'{prefix}_{key}': value for key, value in metrics_dict.items()}
        log_dict["global_step"] = global_step
        wandb.log(log_dict)

    @staticmethod
    def log_images_to_wandb(images, step):
        im_data = []
        column_names = []

        for k, v in images.items():
            column_names.append(k)
            im_data.append(v)

        outputs_table = wandb.Table(data=im_data, columns=column_names)
        wandb.log({f"Step {step} Output": outputs_table})
import os
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ExportBestModelCallback(Callback):
    """Export the best model checkpoint to a fixed location."""
    def __init__(self, export_dir='.', monitor='val/psnr', mode='max'):
        self.export_dir = export_dir
        self.monitor = monitor
        self.mode = mode

    def on_validation_end(self, trainer, pl_module):
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            best_path = trainer.checkpoint_callback.best_model_path
            if os.path.exists(best_path):
                import shutil
                dst = os.path.join(self.export_dir, 'best_model.ckpt')
                shutil.copy(best_path, dst)
                pl_module.print(f"Exported best model to {dst}")


class ExportMetricCallback(Callback):
    """Export training/validation metrics to a JSON file at the end of each epoch."""
    def __init__(self, export_dir='.', monitor='all', best_metric_name='val/psnr',
                 best_mode='max', start_after_epoch=1):
        self.export_dir = export_dir
        self.monitor = monitor
        self.best_metric_name = best_metric_name
        self.best_mode = best_mode
        self.start_after_epoch = start_after_epoch
        self.metrics_history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_after_epoch:
            return
        epoch_metrics = {'epoch': trainer.current_epoch}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, (int, float)):
                epoch_metrics[key] = float(value)
        self.metrics_history.append(epoch_metrics)
        export_path = os.path.join(self.export_dir, 'metrics.json')
        with open(export_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from utils.helper_fncs import prepare_dict_for_yaml
from callbacks.callback import Callback


class PlotMetrics(Callback):
    def __init__(self, nr_epochs, nr_classes=2, zero_class_is_nan_class=True, every_n_steps=None, every_n_epochs=None, log_dir=None, apply_on_out_vars=['out'], **kwargs):
        super().__init__(nr_epochs, every_n_steps, every_n_epochs, log_dir)
        self.callback_type = 'pred_callback' 
        self.zero_class_is_nan_class = zero_class_is_nan_class
        self.nr_classes = nr_classes #Can be used for classification problems or for signal decomposition problems, where it refers to the number of modes. 
        self.apply_on_out_vars = apply_on_out_vars
        
        if log_dir:
            self.log_dir = log_dir / 'metrics'
            (self.log_dir).mkdir(exist_ok=True, parents=False)
        else:
            self.log_dir = log_dir

    def write_to_metric_dict(self, epoch, metrics_dict, metric_name, value, data_fold):
        metric = metrics_dict.get(metric_name, {})
        if not data_fold in metric.keys():
            metric[data_fold] = np.zeros((self.nr_epochs,))*np.nan 
        metric[data_fold][epoch] = value 

        if not metric_name in metrics_dict.keys():
            metrics_dict[metric_name] = metric

        return metrics_dict


    def return_common_part_of_metric_name(self, list_of_names):
        "From a list of metric names, return the common part that most frequently occurs in the list of metrics, to be used to name the saved figure"
        if len(list_of_names) == 1:
            return list_of_names[0]
        else:
            # Split the names in the List on underscores and find out which of the substrings occurred most
            list_of_subnames = list(np.concatenate([name.split("_") for name in list_of_names]))
            return max(set(list_of_subnames), key = list_of_subnames.count)

    def on_epoch_end(self, epoch, metric, name, **kwargs):
        """[summary]

        Args:
            epoch (int): Epoch index
            metric (dict or list): Dict contains for train and val set the specific metric. If list is given, multiple metrics are provided and are plotted in a subplot.
            name (str or list): Name of the metric(s). First name in list will be used to name the saved subplot.
        """
        if self.check_epoch(epoch):    
            super().on_epoch_end(epoch)

            if not isinstance(name, list):
                name = [name]
                assert not isinstance(metric, list)
                metric = [metric]

            if len(name) <= 5: #Only one row of figures
                grid = False
                f, axs = plt.subplots(1,len(name),figsize=(6.4*len(name),4.8))
                if len(name) == 1:
                    axs = np.array([[axs]])
                else:
                    axs = np.array([axs])
            else: #2D grid of figures
                if len(name) == 6: #toy case
                    f, axs = plt.subplots(2,3,figsize=(6.4*2, 4.8*3))
                else:
                    f, axs = plt.subplots(int(np.ceil((len(name)/(self.nr_classes-int(self.zero_class_is_nan_class))))) , int(np.ceil(self.nr_classes-int(self.zero_class_is_nan_class))),figsize=(6.4*(self.nr_classes-int(self.zero_class_is_nan_class)),4.8*(len(name)//(self.nr_classes-int(self.zero_class_is_nan_class)))))
                grid = True

            plt.subplots_adjust(wspace=0.3) 
            
            if grid: min_val, max_val = -0.1, 1.1
            for idx, (met_name, met) in enumerate(zip(name, metric)):           
                marker = 'o' if epoch == 0 else None
                if len(name) == 1:
                    row, col = 0,0
                elif len(name) <= 5:
                    row, col = 0, idx
                else:
                    row, col = np.unravel_index(idx,(int(np.ceil((len(name)/(self.nr_classes-int(self.zero_class_is_nan_class))))),int(np.ceil(self.nr_classes-int(self.zero_class_is_nan_class)))))
                
                axs[row,col].grid()
                for data_fold, vals in met.items():
                    non_nan_vals = vals[:epoch+1][~np.isnan(vals[:epoch+1])]
                    non_nan_epochs = np.arange(1, epoch+2)[~np.isnan(vals[:epoch+1])]
                    axs[row,col].plot(non_nan_epochs, non_nan_vals, marker=marker, label=data_fold)
                axs[row,col].set_xlabel('Epoch')
                axs[row,col].set_ylabel(met_name)
                axs[row,col].legend()
                if grid:
                    axs[row,col].set_ylim([min_val, max_val])

            if self.log_dir is not None:
                try:
                    
                    save_name = self.return_common_part_of_metric_name(name)
                    plt.savefig(self.log_dir/ (save_name +'.png'), bbox_inches='tight')
                except Exception as e:
                    print(e)
            plt.close(f)

    def at_inference(self, metrics_dict, data_fold, **kwargs):
        # Remove the nested structure where each data fold has its own dict, as during inference one metric dict is made per data fold.
        for name, metric in metrics_dict.items():
            if isinstance(metric[data_fold], list) or isinstance(metric[data_fold], tuple):
                metrics_dict[name] = metric[data_fold][0]
            else:
                metrics_dict[name] = metric[data_fold]
        return prepare_dict_for_yaml(metrics_dict)

    def on_train_end(self, metrics_dict, **kwargs):
        with open(self.log_dir / 'all_metrics_training.yml', 'w') as outfile:
                yaml.dump(prepare_dict_for_yaml(metrics_dict), outfile, default_flow_style=False)



class SaveModel(Callback):
    def __init__(self, nr_epochs, save_last=True, every_n_steps=None, every_n_epochs=None, log_dir=None, **kwargs):
        super().__init__(nr_epochs=nr_epochs,  every_n_steps=every_n_steps, every_n_epochs=every_n_epochs, log_dir=log_dir)

        if log_dir:
            self.log_dir = log_dir / 'checkpoints'
            (self.log_dir).mkdir(exist_ok=True, parents=False)
        else:
            self.log_dir = log_dir
        self.save_last = save_last  
        self.best_val_loss = np.inf

    def store_checkpoint(self, metrics_dict, state_dict, optimizer_state_dict, scheduler_state_dict, epoch, which_model, best_checking_metric):
        checkpoint = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': scheduler_state_dict,
            'metrics_dict': metrics_dict,
            'epoch': epoch,
            'best_checking_metric_SaveModelCallback': best_checking_metric, 
        }
        torch.save(checkpoint, self.log_dir / f'{which_model}_model.pt')

    def on_epoch_end(self, state_dict, loss_dict, epoch, optimizer_state_dict, scheduler_state_dict, **kwargs):
        super().on_epoch_end(epoch) 

        if loss_dict['val'][epoch] < self.best_val_loss:

            self.best_val_loss = loss_dict['val'][epoch]
            print(f'Epoch {epoch}: new best model saved.')
            self.store_checkpoint(loss_dict, state_dict, optimizer_state_dict, scheduler_state_dict, epoch, which_model='best', best_checking_metric=self.best_val_loss)

        # Regardless of lowest <checking_metric>, save every <every_n_epochs>
        if self.check_epoch(epoch):
            self.store_checkpoint(loss_dict, state_dict, optimizer_state_dict, scheduler_state_dict, epoch,  which_model=f'epoch{epoch}', best_checking_metric=self.best_val_loss)
        
        if self.save_last:
            # Always overwrite the last stored model to enable continuation of training.
            self.store_checkpoint(loss_dict, state_dict, optimizer_state_dict, scheduler_state_dict, epoch,  which_model=f'last', best_checking_metric=self.best_val_loss)


class EarlyStopper(PlotMetrics):
    def __init__(self, nr_epochs, every_n_steps=1, min_delta=0, patience=1, stop_at_minimum=True, **kwargs):
        super().__init__( nr_epochs=nr_epochs, every_n_steps=every_n_steps)
        self.patience = patience
        self.counter = 0
        self.stop_at_minimum = stop_at_minimum
        self.optimal_checking_loss = np.inf
        self.min_delta = min_delta
        if not self.stop_at_minimum:
            self.optimal_checking_loss *= -1

    def on_epoch_end(self, loss_dict, epoch, **kwargs):

        new_loss = loss_dict['val'][epoch]

        if not np.isnan(new_loss):
            if self.stop_at_minimum:
                if new_loss < (self.optimal_checking_loss - self.min_delta):
                    self.optimal_checking_loss = new_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        return True
            else: # Find the maximum
                if new_loss > (self.optimal_checking_loss + self.min_delta):
                    self.optimal_checking_loss = new_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        return True
            return False

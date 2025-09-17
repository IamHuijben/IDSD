from pathlib import Path

class Callback():

    def __init__(self, nr_epochs, every_n_steps=None, every_n_epochs=None, log_dir=None, **kwargs):

        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.nr_epochs = int(nr_epochs)

        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True, parents=True)

    def on_train_begin(self, **kwargs):
        return

    def on_train_end(self, **kwargs):
        return

    def check_epoch(self, epoch):
        if self.every_n_epochs is None:
            return True
        elif self.every_n_epochs == 0:
            return False
        elif ((epoch % self.every_n_epochs == 0)) or (epoch==self.nr_epochs-1):
            return True
        else:
            return False    

    def check_step(self, step):
        if self.every_n_steps is None:
            return False
        elif (step % self.every_n_steps == 0) and step > 0:
            return True
        else:
            return False

    def on_epoch_begin(self, epoch, **kwargs):
        return

    def on_epoch_end(self, epoch, **kwargs):
        return

    def on_step_begin(self, step, **kwargs):
        return

    def on_step_end(self, step, **kwargs):
        return

    def at_inference(self, epoch, **kwargs):
        return

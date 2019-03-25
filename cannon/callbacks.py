import pickle
import torch
from torch.autograd import Variable
from cannon.utils import cuda_move
import random
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from comet_ml import Experiment


try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    print("tensorboardX is not available. TBCallback cannot be used without it.")

class TrainingCallback:
    def __init__(self):
        pass

    def before_training(self, model_trainer):
        pass

    def after_epoch(self, model_trainer, train_data, validation_data):
        pass

    def after_train_before_validate(self, model_trainer):
        pass

    def after_training(self, model_trainer):
        pass

    def after_training_interrupted(self, model_trainer):
        pass

def save_training_checkpoint(self, e):
    torch.save(self.model, self.log_dir + 'model_e.pt')
    if self.best_vl_metric < self.val_accs[-1]:
        self.best_result = {
            'tr_loss': self.train_losses[-1],
            'tr_acc': self.train_accs[-1],
            'vl_loss': self.val_losses[-1],
            'vl_acc': self.val_accs[-1]
        }
        self.best_vl_metric = self.val_accs[-1]
        self.best_epoch = e
        torch.save(self.model, self.log_dir + 'best_model.pt')
    train_params = self._train_dict()
    d = {
        'model_params': self.model.params_dict(),
        'train_params': train_params,
        'best_result': self.best_result,
        'tr_loss': self.train_losses,
        'vl_loss': self.val_losses,
        'tr_accs': self.train_accs,
        'vl_accs': self.val_accs
    }
    with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
        pickle.dump(d, f)


class TBCallback(TrainingCallback):
    def __init__(self, log_dir, input_dim=None):
        self.log_dir = log_dir
        self.input_dim = input_dim
        self.writer = SummaryWriter(log_dir)
        super().__init__()

    def before_training(self, model_trainer):
        if self.input_dim is not None:
            dummy_input = cuda_move(Variable(torch.zeros(self.input_dim)))
            model_file = self.log_dir + 'onnx_model.proto'
            torch.onnx.export(model_trainer.model, dummy_input, model_file, verbose=True)
            self.writer.add_graph_onnx(model_file)
        pass

    def after_epoch(self, model_trainer, train_data, validation_data):
        n_iter = model_trainer.global_step
        train_loss, train_metric = model_trainer.train_losses[-1], model_trainer.train_metrics[-1]
        val_loss, val_metric = model_trainer.val_losses[-1], model_trainer.val_metrics[-1]

        # data grouping by `slash`
        self.writer.add_scalar('data/train_loss', train_loss, n_iter)
        self.writer.add_scalar('data/train_metric', train_metric, n_iter)
        self.writer.add_scalar('data/val_loss', val_loss, n_iter)
        self.writer.add_scalar('data/val_metric', val_metric, n_iter)

        if n_iter % model_trainer.validation_steps == 0:
            # self.writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
            for name, param in model_trainer.model.named_parameters():
                self.writer.add_histogram('param/' + name, param.clone().cpu().data.numpy(), n_iter, bins='sturges')
            self._save_gradient_histograms(model_trainer, train_data)

    def after_training(self, model_trainer):
        """ Export scalar data to JSON for external processing and
            save final weights as images.
        """
        # for name, param in model_trainer.model.named_parameters():
        #     param = param.data.clone().cpu()
        #     if len(param.size()) == 2:  # images should have size (width, height, channel)
        #         param = param.unsqueeze(2)
        #     elif len(param.size()) == 1:
        #         param = param.unsqueeze(1)
        #         param = param.unsqueeze(2)
        #     self.writer.add_image(name, param, model_trainer.global_step)

        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

    def _save_gradient_histograms(self, model_trainer, train_data):
        # Add gradient norm histogram
        n_iter = model_trainer.global_step
        random_shuffle = list(train_data.get_one_hot_list())
        random.shuffle(random_shuffle)
        for par in model_trainer.model.parameters():
            par.accumulated_grad = []

        n_samples = 100
        for X_i, y_i in random_shuffle[:n_samples]:
            X_data, y_data = cuda_move(X_i), cuda_move(y_i)
            # TODO: backprop through thousand of time steps
            y_out = model_trainer.model.forward(X_data, logits=True)
            loss = F.binary_cross_entropy_with_logits(y_out, y_data)
            model_trainer.model.zero_grad()
            loss.backward()

            for par in model_trainer.model.parameters():
                par.accumulated_grad.append(par.grad)

        for name, par in model_trainer.model.named_parameters():
            t = torch.stack(par.accumulated_grad, 0)
            self.writer.add_histogram('grad/' + name, t.clone().cpu().data.numpy(), n_iter, bins='sturges')
            par.accumulated_grad = None

    def __str__(self):
        return "TBCallback(logdir={})".format(self.log_dir)


class SampleVisualizer(TrainingCallback):
    def __init__(self):
        super().__init__()

    def after_epoch(self, model_trainer, train_data, validation_data):
        for xi, yi in validation_data.iter():
            y_pred = model_trainer.model(xi)
            break
        sample_str = validation_data.visualize_sample(xi[:, 0], y_pred[0])
        model_trainer.logger.info(f"model output  sample: \n{sample_str}")


class LRDecayCallback(TrainingCallback):
    """ Periodically reduce the learning rate if the validation performance is not improving.
    The optimizer used by the trainer must be torch.optim.SGD.

    Attributes:
        decay_rate (float): multiplicative factor used to compute the new lr. must be < 1.
    """
    def __init__(self, decay_rate):
        super().__init__()
        self.scheduler = None
        self.decay_rate = decay_rate
        self.prev_lr = None

    def before_training(self, model_trainer):
        assert type(model_trainer.opt) == torch.optim.SGD
        self.scheduler = ReduceLROnPlateau(model_trainer.opt, 'min', factor=self.decay_rate, verbose=False)
        self.prev_lr = model_trainer.opt.param_groups[0]['lr']

    def after_epoch(self, model_trainer, train_data, validation_data):
        curr_val_loss = model_trainer.val_losses[-1]
        self.scheduler.step(curr_val_loss)
        new_lr = model_trainer.opt.param_groups[0]['lr']
        if self.prev_lr > new_lr:
            self.prev_lr = new_lr
            model_trainer.logger.info("learning rate decreased to {:5e}".format(new_lr))

    def __str__(self):
        return "LRDecayCallback(decay_rate={}))".format(self.decay_rate)


class LearningCurveCallback(TrainingCallback):
    """
        Plot the learning curve for train/validation sets loss and metric and stores it into the log_dir folder.
    """
    def after_epoch(self, model_trainer, train_data, validation_data):
        plot_dir = model_trainer.log_dir + 'plots/'
        os.makedirs(plot_dir, exist_ok=True)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(model_trainer.train_losses, label='train')
        ax.plot(model_trainer.val_losses, label='valid')
        ax.set_title("Loss")
        ax.set_xlabel("#epochs")
        ax.set_xlabel("loss")
        ax.legend()
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure('fig')
        fig.savefig(plot_dir + 'lc_loss.png')
        plt.close(fig)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(model_trainer.train_metrics, label='train')
        ax.plot(model_trainer.val_metrics, label='valid')
        ax.set_title("Metric")
        ax.set_xlabel("#epochs")
        ax.set_xlabel("metric")
        ax.legend()
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure('fig')
        fig.savefig(plot_dir + 'lc_metric.png')
        plt.close(fig)

    def __str__(self):
        return "LearningCurveCallback(TrainingCallback)"


class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, patience):
        super().__init__()
        self.patience = patience
        self._best_loss = -10 ** 9
        self._best_epoch = 0

    def before_training(self, model_trainer):
        self._best_loss = -10 ** 9
        self._best_epoch = 0

    def after_epoch(self, model_trainer, train_data, validation_data):
        e = model_trainer.global_step
        if model_trainer.best_vl_metric >= model_trainer.val_metrics[-1] and \
                e - model_trainer.best_epoch > self.patience:
            model_trainer.logger.info("Early stopping at epoch {}".format(e))
            model_trainer.stop_train()

    def __str__(self):
        return f"EarlyStoppingCallback(patience={self.patience})"


class ModelCheckpoint(TrainingCallback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def after_train_before_validate(self, model_trainer):
        model_name = self.log_dir + 'best_model'
        if os.path.isfile(model_name + '.pt'):
            device = f'cuda:{torch.cuda.current_device()}'
            model_trainer.model = torch.load(model_name + '.pt', device)
            model_trainer.logger.info("Loaded best model checkpoint before final validation.")
        elif os.path.isfile(model_name + '.ptj'):
            device = f'cuda:{torch.cuda.current_device()}'
            model_trainer.model = torch.jit.load(model_name + '.ptj', device)
            model_trainer.logger.info("Loaded best model checkpoint before final validation.")

    def after_epoch(self, model_trainer, train_data, validation_data):
        def try_save(model_name):
            if isinstance(model_trainer.model, torch.jit.ScriptModule):
                # ScriptModule should be checked first because it is a subclass of nn.Module.
                model_trainer.model.save(model_name + '.ptj')
            elif isinstance(model_trainer.model, torch.nn.Module):
                torch.save(model_trainer.model, model_name + '.pt')
            else:
                raise TypeError("Unrecognized model type. Cannot serialize.")

        try:
            try_save(self.log_dir + 'model_e')
            if model_trainer.val_metrics[-1] == max(model_trainer.val_metrics):
                try_save(self.log_dir + 'best_model')
        except Exception as err:
            model_trainer.logger.debug(err)

    def __str__(self):
        return f"ModelCheckpoint(TrainingCallback)"


class CometCallback(TrainingCallback):
    def __init__(self, config_file, tag_list):
        self.tag_list = tag_list
        super().__init__()
        with open(config_file, 'r') as f:
            comet_config = json.load(f)
        comet_exp = Experiment(**comet_config)
        for tag in tag_list:
            comet_exp.add_tag(tag)
        self.exp = comet_exp

    def before_training(self, model_trainer):
        def flatten_dict(d, prefix=''):
            new = {}
            for k,v in d.items():
                if type(v) is dict:
                    new_prefix = prefix + k + '_'
                    flat_v = flatten_dict(v, prefix=new_prefix)
                    new = {**new, **flat_v}
                else:
                    new[prefix + k] = v
            return new
        flat_d = flatten_dict(model_trainer._hyperparams_dict)
        self.exp.log_parameters(flat_d)

    def after_training(self, model_trainer):
        self.exp.end()

    def after_epoch(self, model_trainer, train_data, validation_data):
        self.exp.log_epoch_end(model_trainer.global_step)
        with self.exp.train():
            self.exp.log_metric("loss", model_trainer.train_losses[-1], model_trainer.global_step)
            self.exp.log_metric("metric", model_trainer.train_metrics[-1], model_trainer.global_step)
        with self.exp.validate():
            self.exp.log_metric("loss", model_trainer.val_losses[-1], model_trainer.global_step)
            self.exp.log_metric("metric", model_trainer.val_metrics[-1], model_trainer.global_step)

    def after_training_interrupted(self, model_trainer):
        self.exp.add_tag('keyboard_interrupted')

    def __str__(self):
        str_tags = ", ".join(self.tag_list)
        return f"CometCallback({str_tags})"

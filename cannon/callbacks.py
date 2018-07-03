import pickle
import torch
from torch.autograd import Variable
from cannon.utils import cuda_move
import random
import torch.nn.functional as F

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    print("tensorboardX is not available. TBCallback cannot be used without it.")

class TrainingCallback:
    def __init__(self):
        pass

    def before_training(self, model_trainer):
        pass

    def after_training(self, model_trainer):
        pass

    def after_epoch(self, model_trainer, train_data, validation_data):
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
    train_params = self.train_dict()
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

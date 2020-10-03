import numpy as np
import torch
import time
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F

from quicknat import QuickNat
import utils.common_utils as cu


# Credit to https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
class RunningStats:
    """
    Method to calculate mean and variance in an online manner
    """

    def __init__(self):
        self.n = torch.tensor(0.)
        self.old_m = torch.tensor(0.)
        self.new_m = torch.tensor(0.)
        self.old_s = torch.tensor(0.)
        self.new_s = torch.tensor(0.)

    def clear(self):
        self.n = 0.

    def push(self, x):
        self.n += 1.

        if self.n == 1.:
            self.old_m = self.new_m = x
            self.old_s = 0.
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def get_val(self, metric):
        if metric == 'mean':
            return self.new_m if self.n else 0.
        elif metric == 'variance':
            return self.new_s / (self.n - 1.) if self.n > 1. else 0.
        else:
            raise ValueError('Metric must be either "mean" or "variance"')


class ImportanceModel(QuickNat):
    def __init__(self, params, device, metrics=None, weights=None, featurewise=None, agg_dims=None, normalize=None,
                 supervised=None, uncertainty=None, overwrite_importances=False):
        """
        :param params: Network parameters for QuickNat
        :param device: GPU idx to use
        :param metrics: Metrics to use for the standard and uncertainty importances
        :param weights: Weighting of the standard and uncertainty importances
        :param featurewise: Specifies if importances should be averaged featurewise
        :param normalize: Specifies if importances should be normalize to interval between 0 and 1
        """
        super(ImportanceModel, self).__init__(params)
        self.device = device
        self.metrics, self.weights, self.featurewise, self.agg_dims, self.normalize, self.supervised, self.uncertainty = metrics, weights, featurewise, agg_dims, normalize, supervised, uncertainty
        self.overwrite_importances = overwrite_importances
        self.old_parameters = {}
        self.freeze_masks = {}
        self.running_importances = {}
        self.running_unc_importances = {}
        self.running_uncertainty = None

    def update_old_parameters(self):
        """
        Saves a copy of the parameters which don't get changed that are necessary to calculate the surrogate loss
        """
        self.old_parameters = {n: p.clone().detach().cuda(self.device) for n, p in self.named_parameters() if
                               p.requires_grad}

    def update_freeze_masks(self, freeze_cap, dynamic_fixing=True):
        if self.freeze_masks == {}:
            self.freeze_masks = {n: torch.ones_like(p) for n, p in self.named_parameters() if p.requires_grad}

        importances = self.get_importances()

        if dynamic_fixing:
            percentile = np.percentile(torch.cat([importance.flatten() for importance in importances.values()]).cpu(),
                                       100 - freeze_cap)
        else:
            percentile = self.get_percentile_fixed_capacity(importances, freeze_cap)

        if percentile:
            for name, param in self.freeze_masks.items():
                self.freeze_masks[name] = param * (importances[name] < percentile).type(torch.float)

    def update_rnd_freeze_masks(self, freeze_method, freeze_total_cap, filter_level):
        conv_layers = {name: param for name, param in self.named_parameters() if 'conv' in name and 'weight' in name}

        if self.freeze_masks == {}:
            self.freeze_masks = {name: torch.rand_like(param) for name, param in conv_layers.items()}

        module_namess = [name[:-7].split('.') for name in
                         [name for name, _ in self.named_parameters() if 'conv' in name and 'weight' in name]]

        num_layers = {names[0]: 0 for names in module_namess}
        for names in module_namess:
            module = self
            for name in names:
                module = module._modules[name]
            num_layers[names[0]] += module.out_channels * module.in_channels if filter_level else np.prod(
                module.weight.size())

        if freeze_method == 'equally':
            freeze_caps = {name: freeze_total_cap / 100 * num_layer for name, num_layer in num_layers.items()}
        else:
            L = len(num_layers)
            if freeze_method == '-linearly':
                m = (2 * freeze_total_cap / 100 * sum(num_layers.values())) / (-(L * L) + L)
                t = -L * m
            else:
                m = (2 * freeze_total_cap / 100 * sum(num_layers.values())) / (L * L + L)
                t = 0

            freeze_caps = {name: m * i + t for i, name in enumerate(num_layers.keys(), 1)}

        freeze_caps = {name: min(cap, num_layers[name]) for name, cap in freeze_caps.items()}

        for i, (name, param) in enumerate(self.freeze_masks.items()):
            layer_name = name.split('.')[0]
            if filter_level and freeze_caps[layer_name] != 0:
                freeze_cap = int(np.prod(param.size()[:2]) / num_layers[layer_name] * freeze_caps[layer_name])
                sum_filters = param.sum((2, 3))
                sum_filters_flat = sum_filters.flatten().cpu().numpy()
                val = sum_filters_flat[
                    np.nonzero(np.argpartition(sum_filters_flat, freeze_cap - 1) == freeze_cap - 1)[0][0]]
                freeze_filter_idxs = np.argwhere(sum_filters.cpu().numpy() <= val)

                for filter_idx in freeze_filter_idxs:
                    self.freeze_masks[name][tuple(filter_idx)].fill_(0.)
            else:
                freeze_cap = freeze_caps[layer_name] / num_layers[layer_name]
                self.freeze_masks[name][param < freeze_cap] = 0.

    def calc_importances(self, dataset, loss_func, metric):
        """
        Calculates the models weight importances
        :param dataset: Dataset to iterate over and calculate the gradients
        :param loss_func: If not given, importances are calculated in an unsupervised fashion using the L2 distance
        :param metric: Metric that specifies gradients form
        """
        if self.running_importances == {} or self.overwrite_importances:
            self.running_importances = {n: RunningStats() for n, p in self.named_parameters() if p.requires_grad}

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        mode = self.training
        self.eval()

        curr_parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}

        start = time.time()
        for i, (_, imgs, labels, class_weights) in enumerate(dataloader):
            img = imgs[0].unsqueeze(dim=0)
            if torch.cuda.is_available():
                img = img.type(torch.FloatTensor).cuda(self.device)
                labels = labels.type(torch.LongTensor).cuda(self.device)
                class_weights = class_weights.type(torch.FloatTensor).cuda(self.device)

            output = self.forward(img)

            loss = loss_func(output, labels, class_weights) if loss_func else torch.norm(F.softmax(output, dim=1)).pow(
                2)

            self.zero_grad()
            loss.backward()

            for name, rs in self.running_importances.items():
                self.running_importances[name].push(
                    curr_parameters[name].grad.abs() if metric == 'mean' else curr_parameters[name].grad)

            cu.print_progress(start, i, len(dataloader), 'Sample: [{} / {}]'.format(i, len(dataloader)))
        self.train(mode=mode)

    def calc_online_importances(self, metric):
        if self.running_importances == {} or self.overwrite_importances:
            self.running_importances = {n: RunningStats() for n, p in self.named_parameters() if p.requires_grad}

        curr_parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}

        for name, rs in self.running_importances.items():
            self.running_importances[name].push(
                curr_parameters[name].grad.abs() if metric == 'mean' else curr_parameters[name].grad)

    def calc_uncertainty_importances(self, dataset, batch_size, num_mc_samples):
        if self.running_unc_importances == {}:
            self.running_unc_importances = {n: RunningStats() for n, p in self.named_parameters() if p.requires_grad}

        self.running_uncertainty = RunningStats()

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        mode = self.training
        self.eval()

        self.enable_test_dropout()

        curr_parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}

        start = time.time()
        for i, (_, imgs, _, _) in enumerate(dataloader):
            img = imgs[0].unsqueeze(dim=0)
            if torch.cuda.is_available():
                img = img.type(torch.FloatTensor).cuda(self.device)

            div, mod = divmod(num_mc_samples, batch_size)
            iterations = div + 1 if mod != 0 else div
            for j in range(iterations):
                imgs = img.repeat(min(batch_size, num_mc_samples - j * batch_size), 1, 1, 1)
                outputs = self.forward(imgs)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                # as 0 * log(0) returns nan we add a small value to the outputs to avoid this issue
                outputs = outputs + 1e-10
                if j == 0:
                    uncertainty_per_class = -(outputs * torch.log(outputs)).sum(0)
                else:
                    uncertainty_per_class += -(outputs * torch.log(outputs)).sum(0)

            uncertainty = uncertainty_per_class.sum(0).mean()
            self.running_uncertainty.push(uncertainty)
            self.zero_grad()
            uncertainty.backward()

            for name, rs in self.running_unc_importances.items():
                self.running_unc_importances[name].push(curr_parameters[name].grad.abs())

            cu.print_progress(start, i, len(dataloader), 'Sample: [{} / {}]'.format(i, len(dataloader)))
        self.train(mode=mode)

    def get_importances(self):
        """
        Helper function to get plain importances based on defined methods. The higher the more important a value
        :return: Plain importances as tensors
        """
        importancess = [{name: importance.get_val(metric) for name, importance in running.items()} for running, metric
                        in zip([imp for imp in [self.running_importances, self.running_unc_importances] if imp != {}],
                               self.metrics)]

        mean_uncertainty = None if self.running_uncertainty is None else self.running_uncertainty.get_val('mean')

        if self.normalize:
            for importances in importancess:
                importances_flat = torch.cat([imp.flatten() for imp in importances.values()])

                if self.normalize == 'standard':
                    min_val = torch.min(importances_flat)
                    max_val = torch.max(importances_flat)

                    for name, importance in importances.items():
                        importances[name] = (importance - min_val) / (max_val - min_val)
                else:
                    q1, q3 = np.percentile(importances_flat.cpu().numpy(), [5, 95])
                    iqr = q3 - q1
                    lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)

                    importances_clamped_log = {name: torch.log(torch.clamp(imp, min=lower_bound, max=upper_bound)) for
                                               name, imp in importances.items()}

                    importances_clamped_log_flat = torch.cat(
                        [imp.flatten() for imp in importances_clamped_log.values()])

                    min_val = torch.min(importances_clamped_log_flat)
                    max_val = torch.max(importances_clamped_log_flat)

                    for name, importance in importances_clamped_log.items():
                        importances[name] = (importance - min_val) / (max_val - min_val)

        if self.featurewise:
            for importances in importancess:
                for name, importance in importances.items():
                    if 'conv' in name and 'weight' in name:
                        if self.featurewise == 'avg':
                            agg = torch.mean(importance, (2, 3) if self.agg_dims == 2 else (1, 2, 3), keepdim=True)
                        elif self.featurewise == 'max':
                            if self.agg_dims == 2:
                                agg = \
                                    torch.max(torch.reshape(importance, (*importance.size()[:2], -1)), -1,
                                              keepdim=True)[
                                        0].unsqueeze(3)
                            else:
                                agg = \
                                    torch.max(torch.reshape(importance, (*importance.size()[:1], -1)), -1,
                                              keepdim=True)[
                                        0].unsqueeze(2).unsqueeze(3)

                        importances[name] = agg.repeat(
                            1, 1, *importance.size()[-2:]) if self.agg_dims == 2 else agg.repeat(1, *importance.size()[
                                                                                                     -3:])

        for importances, metric in zip(importancess, self.metrics):
            for name, importance in importances.items():
                if self.supervised:
                    if self.normalize:
                        importances[name] = 1 - importance
                    else:
                        importances[name] = 1 / importance
                elif self.uncertainty:
                    importances[name] = 1 / (mean_uncertainty + importance.abs())
                else:
                    if self.normalize:
                        importances[name] = 1 - importance if metric == 'variance' else importance
                    else:
                        importances[name] = 1 / importance if metric == 'variance' else importance

        return {name: sum([weight * importances[name] for importances, weight in zip(importancess, self.weights)]) for
                name in importancess[0].keys()}

    def surrogate_loss(self, average=False):
        """
        Calculates the additional loss for regularization
        :return: Surrogate loss
        """
        loss = 0.0
        importances = self.get_importances()
        for n, p in self.named_parameters():
            if p.requires_grad:
                new_loss = torch.sum(importances[n] * (self.old_parameters[n] - p).pow(2))
                loss += new_loss
        if average:
            num_params = sum([np.prod(imp.size()) for imp in importances.values()])
            return loss / num_params
        else:
            return loss

    def get_percentile_fixed_capacity(self, importances, capacity):
        remaining_imps = torch.cat(
            [imp[mask != 0] for imp, mask in zip(importances.values(), self.freeze_masks.values())])

        if len(remaining_imps) == 0:
            return None

        fixed_cap = len(remaining_imps) / sum([np.prod(imp.size()) for imp in importances.values()])

        cap_to_fix = capacity / fixed_cap

        return np.percentile(remaining_imps.cpu(), 100 - cap_to_fix)

def freeze_weights_besides(model, besides_names):
    """
    Freezes weights of network beside specific ones
    :param besides_names: Determines names of parameters that should not be freezed
    """

    for name, param in model.named_parameters():
        if len([b_name for b_name in besides_names if b_name in name]) == 0:
            param.requires_grad_(False)


# Implementation of metrics proposed in GEM: https://arxiv.org/pdf/1706.08840.pdf
def calc_avg_acc(resultss):
    return sum(resultss[-1]) / len(resultss)


def calc_bwt(resultss):
    bwt = sum([resultss[-1][i] - resultss[i][i] for i in range(len(resultss))]) / (len(resultss) - 1)
    return 1 - bwt.applymap(lambda x: abs(min(x, 0)))


# Implementation of metrics proposed in https://arxiv.org/abs/1810.13166
def calc_cl_acc(resultss):
    """
    Calculates the average accuracy of a CL sequence of tasks
    :param resultss: Matrix containing identical pandas DataFrames in each cell. Rows contain evaluations for one task
    :return: Average accuracy
    """
    sum_dfs = resultss[0][0].copy() * 0
    for i, task in enumerate(resultss):
        for j, eval in enumerate(task):
            if i >= j:
                sum_dfs += eval.copy()
    return sum_dfs / ((len(resultss) * (len(resultss) + 1)) / 2)


def calc_cl_backw_tf(resultss):
    """
    Calculates the overall Backward Transfer of a CL sequence of tasks, which are seperated in two metrics
    :param resultss: Matrix containing identical pandas DataFrames in each cell. Rows contain evaluations for one task
    :return: Remembering: Negative backward transfer; BWT+: positive backward transfer: improvement over time
    """
    sum_dfs = resultss[0][0].copy() * 0
    sum_dfs_rem = resultss[0][0].copy() * 0
    sum_dfs_backw_tf_plus = resultss[0][0].copy() * 0
    for i in range(1, len(resultss)):
        for j in range(i):
            sum_dfs += resultss[i][j] - resultss[j][j]
            sum_dfs_rem += 1 - (resultss[i][j] - resultss[j][j]).applymap(lambda x: abs(min(x, 0)))
            sum_dfs_backw_tf_plus += (resultss[i][j] - resultss[j][j]).applymap(lambda x: max(x, 0))

    backw_tf = sum_dfs / ((len(resultss) * (len(resultss) - 1)) / 2)
    remembering = sum_dfs_rem / ((len(resultss) * (len(resultss) - 1)) / 2)
    backw_tf_plus = sum_dfs_backw_tf_plus / ((len(resultss) * (len(resultss) - 1)) / 2)

    return backw_tf, remembering, backw_tf_plus


def calc_cl_fw_tf(resultss):
    """
    Calculates the influence that learning a task has on the performance of future tasks
    :param resultss: Matrix containing identical pandas DataFrames in each cell. Rows contain evaluations for one task
    :return: Forward Transfer
    """
    sum_dfs = resultss[0][0].copy() * 0
    for i, task in enumerate(resultss):
        for j, eval in enumerate(task):
            if i < j:
                sum_dfs += eval.copy()

    return sum_dfs / ((len(resultss) * (len(resultss) - 1)) / 2)


def calc_transfer_learning_acc(resultss):
    """
    Calculates the average accuracy of all tasks
    :param resultss: Matrix containing identical pandas DataFrames in each cell. Rows contain evaluations for one task
    :return: Average accuracy
    """
    sum_dfs = resultss[0][0].copy() * 0
    for i, task in enumerate(resultss):
        for j, eval in enumerate(task):
            if i == j:
                sum_dfs += eval.copy()
    return sum_dfs / len(resultss)


class ParamSpecificAdam(torch.optim.Adam):
    """
    Custom implementation of Adam that can handle parameter-specific learning rates
    """

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Changed p.data.addcdiv_(-step_size, exp_avg, denom) to
                p.data.add_(-step_size * (exp_avg / denom))

        return loss


class ParamSpecificSGD(torch.optim.SGD):

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data = p.data - group['lr'] * d_p

        return loss

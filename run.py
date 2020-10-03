import os
import argparse
import torch
import pandas as pd
from glob import glob
from shutil import copyfile
import re
from torch.utils.data import DataLoader

from utils.settings import Settings
from utils import data_utils as du, losses as additional_losses
from utils import common_utils as cu
from quicknat import QuickNat
from importance_model import ImportanceModel, ParamSpecificAdam, ParamSpecificSGD, calc_cl_acc, calc_cl_backw_tf, \
    calc_cl_fw_tf, calc_transfer_learning_acc, freeze_weights_besides
from solver import Solver
from utils.log_utils import LogWriter
import utils.evaluator as eu


def train(train_params, optim_params, common_params, data_params, net_params, importance_params=None):
    if not importance_params:
        importance_params = {}

    train_data = du.get_dataset(data_params['data_dir'], 'train', sliced=True)
    val_data = du.get_dataset(data_params['data_dir'],
                              'val' if 'val_data' in data_params.keys() and data_params['val_data'] else 'test',
                              sliced=True)

    train_loader = DataLoader(train_data, batch_size=train_params['train_batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=train_params['val_batch_size'], shuffle=False, num_workers=4,
                            pin_memory=True)

    # Load model
    if train_params['use_pre_trained']:
        model = torch.load(train_params['pre_trained_path'])

    else:
        if common_params['model_name'] == 'quicknat':
            model = QuickNat(net_params)
        elif common_params['model_name'] == 'importances_quicknat':
            model = ImportanceModel(net_params, common_params['device'],
                                    None if 'metrics' not in importance_params.keys() else importance_params['metrics'],
                                    None if 'weights' not in importance_params.keys() else importance_params['weights'],
                                    None if 'featurewise' not in importance_params.keys() else importance_params[
                                        'featurewise'],
                                    None if 'agg_dims' not in importance_params.keys() else importance_params[
                                        'agg_dims'],
                                    None if 'normalize' not in importance_params.keys() else importance_params[
                                        'normalize'],
                                    'supervised' in importance_params.keys() and importance_params['supervised'],
                                    'num_mc_samples' in importance_params.keys() and importance_params[
                                        'num_mc_samples'] != 0,
                                    'overwrite_importances' in importance_params.keys() and importance_params[
                                        'overwrite_importances'])
        else:
            raise ValueError('Invalid model architecture defined')

    # Disable batchnorm if necessary
    if 'batchnorm' in net_params and not net_params['batchnorm']:
        model.disable_batchnorm()

    # Create optimizer
    if 'param_specific_lr' in importance_params.keys() and importance_params['param_specific_lr']:
        importances = model.get_importances()
        optim_params['params'] = []

        initial_lr = optim_params['lr'] * (
            1 if 'lr_multiplier' not in importance_params.keys() else importance_params['lr_multiplier'])

        for name, param in model.named_parameters():
            if name not in importances.keys():
                param_lr = torch.full_like(param, optim_params['lr'])
            else:
                param_lr = (1 - importances[name]) ** (
                    2 if 'squared_imp_inv' in importance_params.keys() and importance_params[
                        'squared_imp_inv'] else 1) * initial_lr

            optim_params['params'].append({'params': param, 'lr': torch.clamp(param_lr, 1e-7, 0.99)})

        if train_params['optimizer'] == 'Adam':
            optimizer = ParamSpecificAdam(**optim_params)
        elif train_params['optimizer'] == 'SGD':
            optimizer = ParamSpecificSGD(**optim_params)

    else:
        if train_params['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **optim_params)
        elif train_params['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **optim_params)
        else:
            raise ValueError('Invalid optimizer specified')

    # Create solver and start to train
    solver = Solver(model,
                    device=common_params['device'],
                    optim=optimizer,
                    model_name=common_params['model_name'],
                    exp_name=common_params['exp_name'],
                    classes=data_params['classes'],
                    num_classes=net_params['num_class'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=None if 'lr_scheduler_step_size' not in train_params.keys() else
                    train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=None if 'lr_scheduler_gamma' not in train_params.keys() else train_params[
                        'lr_scheduler_gamma'],
                    save_best_ckpt=train_params['save_best_ckpt'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    checkpoint_path=None if 'checkpoint_path' not in train_params.keys() else train_params[
                        'checkpoint_path'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'],
                    surrogate_reg_param=importance_params[
                        'surrogate_reg_param'] if 'surrogate_reg_param' in importance_params.keys() else 0,
                    average_weight_shifts=False if 'average_weight_shifts' not in importance_params.keys() else
                    importance_params['average_weight_shifts'])

    solver.train(train_loader, val_loader)

    # Calculate importances if needed
    if common_params['model_name'] == 'importances_quicknat':
        if 'surrogate_reg_param' in importance_params.keys():
            model.update_old_parameters()
        if 'num_mc_samples' in importance_params.keys() and importance_params['num_mc_samples'] != 0:
            print('Calculating weight importances...')
            model.calc_uncertainty_importances(val_data, train_params['val_batch_size'],
                                               importance_params['num_mc_samples'])
        if 'supervised' in importance_params.keys():
            print('Calculating weight importances...')
            if 'online' in importance_params.keys() and importance_params['online'] and importance_params['supervised']:
                model.calc_online_importances(importance_params['metrics'][0])
            else:
                model.calc_importances(val_data,
                                       additional_losses.CombinedLoss() if importance_params['supervised'] else None,
                                       importance_params['metrics'][0])
        if 'fixed_capacity' in importance_params.keys() and importance_params['fixed_capacity']:
            model.update_freeze_masks(importance_params['fixed_capacity'],
                                      'dynamic_fixing' in importance_params.keys() and importance_params[
                                          'dynamic_fixing'])

        if 'freeze_method' in importance_params.keys():
            print('Freezing parameters...')
            model.update_rnd_freeze_masks(importance_params['freeze_method'], importance_params['freeze_cap'],
                                          importance_params['filter_level'])

    if common_params['model_name'] == 'bn_quicknat':
        freeze_weights_besides(model, ['batchnorm'])
    elif common_params['model_name'] == 'se_quicknat':
        freeze_weights_besides(model, ['SELayer'])

    # Save model
    model.save(common_params['save_model_dir'], common_params['final_model_file'])
    print('Final model saved at {}'.format(
        os.path.join(common_params['save_model_dir'], common_params['final_model_file'])))


def evaluate(eval_params, data_params, common_params):
    print('Evaluating...')

    classes = data_params['classes']
    log_writer = LogWriter(['dice_scores'], common_params['log_dir'], common_params['exp_name'])

    model = torch.load(eval_params['eval_model_path'])

    if 'classifier_model_path' in eval_params.keys() and eval_params['classifier_model_path'] != eval_params[
        'eval_model_path']:
        classifier_model = torch.load(eval_params['classifier_model_path'])
        model.classifier = classifier_model.classifier

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda(common_params['device'])

    model.eval()

    val_data = du.get_dataset(data_params['data_dir'], 'test', sliced=False)

    volumes_dice_scores = eu.calc_validation_dice_score(model, val_data, classes, eval_params['val_batch_size'],
                                                        common_params['device'])

    df = log_writer.create_df(val_data, volumes_dice_scores, classes, data_params['group_names'], data_params['groups'])

    for col in df.columns:
        log_writer.plot_charts('dice_scores', 'vol_scores/{}'.format(col), df.loc[:, col].to_frame())
    log_writer.plot_charts('dice_scores', 'per_group', df.mean(level=0))
    log_writer.plot_charts('dice_scores', 'per_class', df.mean(level=1))

    df.to_pickle(os.path.join(common_params['log_dir'], common_params['exp_name'], 'dice_score_summary.pkl'))

    log_writer.close()
    print('Mean dice score of validation data: {}'.format(df.mean().mean()))

    return df


def copy_last_eval(settings_name, log_dir):
    permutation, task = [(settings_name.split(start))[1].split('_')[0] for start in ['permut', 'task']]
    domain = (settings_name.split('task' + task + '_'))[1].split('.ini')[0]
    method = (settings_name.split(permutation))[1].split('task')[0]
    order = permutation[:int(task)]
    finished_evals = [eval for eval in glob(os.path.join(log_dir, '*')) if 'CL_Results' not in eval and method in eval]

    for eval in finished_evals:
        eval_name = os.path.basename(eval)
        eval_permutation, eval_task = [(eval_name.split(start))[1].split('_')[0] for start in ['permut', 'task']]
        eval_domain = (settings_name.split('task' + task + '_'))[1].split('.ini')[0]
        eval_method = (settings_name.split(permutation))[1].split('task')[0]
        eval_order = eval_permutation[:int(eval_task)]

        if method == eval_method and order == eval_order and domain == eval_domain:
            summaries = glob(os.path.join(eval, 'dice_score_summary.pkl'))
            if not len(summaries) == 1:
                raise FileNotFoundError('Finished eval file could not be found')
            else:
                summary = summaries[0]
            new_eval_dir = os.path.join(log_dir, settings_name)
            cu.create_if_not(new_eval_dir)
            copyfile(summary, os.path.join(new_eval_dir, os.path.basename(summary)))
            return True
    return False


def copy_last_checkpoint(settings_name, experiment_dir, experiment_name):
    permutation, task = [(settings_name.split(start))[1].split('_')[0] for start in ['permut', 'task']]
    method = (settings_name.split(permutation))[1].split('task')[0]
    order = permutation[:int(task)]
    finished_exps = [exp for exp in glob(os.path.join(experiment_dir, '*')) if
                     'CL_Results' not in exp and method in exp]

    for exp in finished_exps:
        exp_name = os.path.basename(exp)
        exp_permutation, exp_task = [(exp_name.split(start))[1].split('_')[0] for start in ['permut', 'task']]
        exp_method = (settings_name.split(permutation))[1].split('task')[0]
        exp_order = exp_permutation[:int(exp_task)]

        if method == exp_method and order == exp_order:
            checkpoints = glob(os.path.join(exp, 'checkpoints', '*.pth.tar'))

            if not len(checkpoints) == 1:
                raise FileNotFoundError('Finished exp file could not be found')
            else:
                last_checkpoint = checkpoints.sort()[-1]

            new_exp_dir = os.path.join(experiment_dir, experiment_name, 'checkpoints')
            cu.create_if_not(new_exp_dir)
            copyfile(last_checkpoint, os.path.join(new_exp_dir, os.path.basename(last_checkpoint)))
            return True
    return False


def evaluate_cl(common_params, eval_params):
    writers = ['dice_scores', 'cl_accuracy', 'backw_tf', 'remembering', 'backw_tf_plus', 'forward_transfer',
               'transfer_learning']

    experiments = list(set(
        [re.sub(r'_task\d+', '', os.path.basename(path)[os.path.basename(path).index('exp_') + 4:-8]) for path in
         glob(os.path.join(common_params['models_dir'], '*pth.tar'))]))

    metrics = {}
    metrics_agg = {}
    for experiment in experiments:
        log_writer = LogWriter(writers, common_params['log_dir'], os.path.join(common_params['eval_name'], experiment))
        resultss = []

        for model_path in sorted(
                glob(os.path.join(common_params['models_dir'], '*exp_{}_{}*pth.tar'.format(experiment, 'task')))):
            results = []

            model = torch.load(model_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                model.cuda(common_params['device'])
            model.eval()

            for dataset in eval_params['datasets']:
                data_dir = eval_params['data_dir'].format(dataset)
                val_data = du.get_dataset(data_dir, 'val' if 'val_data' in eval_params.keys() and eval_params[
                    'val_data'] else 'test', sliced=False)
                volumes_dice_scores = eu.calc_validation_dice_score(model, val_data, eval_params['classes'],
                                                                    eval_params['batch_size'], common_params['device'])
                df = log_writer.create_df(val_data, volumes_dice_scores, eval_params['classes'],
                                          eval_params['group_names'], eval_params['groups'])
                results.append(df)
            resultss.append(results)

        for task_idx in range(len(resultss)):
            for domain_idx in range(len(resultss)):
                resultss[task_idx][domain_idx].to_csv(
                    os.path.join(common_params['log_dir'], common_params['eval_name'],
                                 'exp-{}_task{}_domain{}.csv'.format(experiment, task_idx + 1, domain_idx + 1)))

        resultss_agg = [[df.mean(axis=1).to_frame() for df in results] for results in resultss]
        log_writer.plot_cl_results('dice_scores', 'results', resultss_agg, eval_params['datasets'], volumes_agg=False)

        metrics[experiment] = [calc_cl_acc(resultss_agg), *calc_cl_backw_tf(resultss_agg), calc_cl_fw_tf(resultss_agg),
                               calc_transfer_learning_acc(resultss_agg)]
        for writer, df in zip(writers[1:], metrics[experiment]):
            log_writer.plot_charts(writer, 'per_group', df.mean(level=0))
            log_writer.plot_charts(writer, 'per_class', df.mean(level=1))
        log_writer.close()

        metrics_agg[experiment] = [float(metric.mean()) for metric in metrics[experiment]] + [float(metric.std()) for
                                                                                              metric
                                                                                              in metrics[experiment]]

    for experiment in experiments:
        for df, col in zip(metrics[experiment], writers[1:]):
            df.columns = [col]

    metrics_combined = {experiment: pd.concat([df.droplevel(0) for df in dfs], axis=1) for experiment, dfs in
                        metrics.items()}

    for experiment, df in metrics_combined.items():
        df['Experiment'] = [experiment for _ in range(df.shape[0])]
        df.to_csv(
            os.path.join(common_params['log_dir'], common_params['eval_name'], 'experiment_{}.csv'.format(experiment)))

    df = pd.DataFrame.from_dict(metrics_agg, orient='index',
                                columns=['mean_' + col for col in writers[1:]] + ['std_' + col for col in writers[1:]])
    df.to_csv(os.path.join(common_params['log_dir'], common_params['eval_name'], 'experiment_summary.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=False, choices=['train', 'eval', 'eval_cl'],
                        help='Run only "train", "eval", "eval_cl" or all experiments if flag is not set')
    parser.add_argument('--run_all_in', '-rai', nargs='*', help='Run all settings inside the given folders')
    parser.add_argument('--settings_paths', '-sp', required=False, nargs='*',
                        help='Paths to settings file. Multiple paths will be trained/evaluated sequently')
    args = parser.parse_args()

    if args.settings_paths:
        settings_paths = args.settings_paths
    else:
        settings_dirs = [item for sublist in
                         [sorted(glob(os.path.join(dir, '**/settings'), recursive=True)) for dir in args.run_all_in] for
                         item in sublist]

        settings_dirs = [dir for dir in settings_dirs if
                         not os.path.exists(os.path.join(os.path.dirname(dir), 'evals'))]

        dirs = [item for pair in
                zip(settings_dirs, [os.path.join(os.path.dirname(path), 'eval_settings') for path in settings_dirs]) for
                item in pair]

        settings_paths = [item for sublist in [sorted(glob(os.path.join(dir, '*.ini'))) for dir in dirs] for item in
                          sublist]

    for settings_path in settings_paths:
        # Make processess deterministic
        torch.manual_seed(34)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        settings = Settings(settings_path)

        if settings['COMMON']['mode'] == 'train':
            settings_name = os.path.basename(settings_path)

            train(settings['TRAINING'], settings['OPTIMIZER'], settings['COMMON'], settings['DATA'],
                  settings['NETWORK'], settings['IMPORTANCES'] if 'IMPORTANCES' in settings.keys() else None)
        elif settings['COMMON']['mode'] == 'eval':
            evaluate(settings['EVAL'], settings['DATA'], settings['COMMON'])
        elif settings['COMMON']['mode'] == 'eval_cl':
            evaluate_cl(settings['COMMON'], settings['EVAL'])

    if args.run_all_in:
        print('FINISHED experiments in {}'.format(args.run_all_in))

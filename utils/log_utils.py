import itertools
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter


class LogWriter(object):
    def __init__(self, writer_names, log_dir_name, exp_name, use_last_checkpoint=False, cm_cmap=plt.cm.Blues):
        self.log_dir_name, self.exp_name, self.cm_cmap = log_dir_name, exp_name, cm_cmap

        if not use_last_checkpoint:
            for name in writer_names:
                if os.path.exists(os.path.join(log_dir_name, exp_name, name)):
                    shutil.rmtree(os.path.join(log_dir_name, exp_name, name))

        self.writers = dict(
            zip(writer_names, [SummaryWriter(os.path.join(log_dir_name, exp_name, name)) for name in writer_names]))

    def plot_cm(self, writer, caption, cm, step=None):
        cm_data = np.array([list(class_dice_scores.values()) for class_dice_scores in cm.values()])

        fig = plt.figure(figsize=(len(cm) // 4, len(cm) // 4), dpi=180, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(cm_data, interpolation='nearest', cmap=self.cm_cmap)
        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(np.arange(len(cm)))
        ax.set_xticklabels(list(cm.keys()), fontsize=4, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(np.arange(len(cm)))
        ax.set_yticklabels(list(cm.keys()), fontsize=4, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        thresh = cm_data.max() / 2.
        for i, j in itertools.product(range(cm_data.shape[0]), range(cm_data.shape[1])):
            ax.text(j, i, format(cm_data[i, j], '.2f') if cm_data[i, j] != 0 else '.', horizontalalignment='center',
                    fontsize=6, verticalalignment='center', color='white' if cm_data[i, j] > thresh else 'black')

        fig.set_tight_layout(True)
        np.set_printoptions(precision=2)

        self.writers[writer].add_figure(caption, fig, step)

    def plot_charts(self, writer, tag, df, step=None):
        if not isinstance(df.index, pd.core.index.MultiIndex):
            index = pd.MultiIndex.from_tuples(zip(['dummy' for _ in range(len(df))], df.index),
                                              names=['D', df.index.name])
            df = df.set_index(index)
        num_df_cols = len(df.columns)
        df = df.iloc[:, 0] if num_df_cols == 1 else df
        idx_names = df.index.get_level_values(0).unique()
        num_cols = [len(df.loc[idx]) for idx in idx_names]
        fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), gridspec_kw={'width_ratios': num_cols}, dpi=300,
                                 figsize=(max(6.4, len(df) // 4), 4.8))
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
        plt.tight_layout(rect=[0, 0.2, 1, 1])
        for i, (ax, idx_name) in enumerate(zip(axes, idx_names)):
            values = df.loc[idx_name]
            ax.set_ylim(0., 1.)

            if idx_name != 'dummy':
                ax.set_xlabel(idx_name, fontsize=10)
                ax.xaxis.set_label_position('top')

            if num_df_cols == 1:
                if i != 0:
                    ax.get_yaxis().set_visible(False)
                ax.set_xticks(np.arange(len(values)))
                rects = ax.bar(np.arange(len(values)), values)
                hline = ax.axhline(values.mean(), color='red')
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), ha='center',
                                va='bottom', fontsize='x-small')
            else:
                bp_dict = ax.boxplot(values, meanline=True, showmeans=True)
                hline = ax.axhline(values.mean().mean(), color='red')
                for line in bp_dict['means']:
                    x, y = np.mean(line.get_xydata()[:, 0]), line.get_xydata()[0, 1]
                    ax.annotate('{:.2f}'.format(y), (x, 10), xycoords=('data', 'axes pixels'), ha='center',
                                fontsize='x-small')

            hline_x, hline_y = hline.get_xydata()[1]
            ax.annotate('{:.2f}'.format(hline_y), (60, hline_y), xycoords=('axes pixels', 'data'), ha='center',
                        va='bottom',
                        fontsize='medium', color='red')
            ax.set_xticklabels(values.index, fontsize=6, rotation=-90, ha='center')
            ax.xaxis.tick_bottom()

            self.writers[writer].add_figure(tag, fig, step)

    def plot_cl_results(self, writer, caption, resultss, datasets, volumes_agg=True):
        fig, axes = plt.subplots(nrows=len(resultss), ncols=len(resultss[0]), dpi=300)

        for ax, dataset in zip(axes[0], datasets):
            ax.set_title(dataset)

        for ax, dataset in zip(axes[:, 0], datasets):
            ax.set_ylabel(dataset, rotation=0, size='large', labelpad=20)

        for i, results in enumerate(resultss):
            for j, result in enumerate(results):
                ax = axes[i][j]
                ax.set_facecolor('lightgrey' if j > i else 'lightblue' if i == j else '#edd9ff')
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
                for spine in ax.spines.values():
                    spine.set_color('white')

                ax.set_ylim(0., 1.)

                bp_dict = ax.boxplot(result.mean() if volumes_agg else result.mean(level=1).mean(axis=1), meanline=True,
                                     showmeans=True)
                for line in bp_dict['means']:
                    x, y = np.mean(line.get_xydata()[:, 0]), line.get_xydata()[0, 1]
                    ax.annotate('{:.3f}'.format(y), (60, 20), xycoords='axes pixels', ha='center',
                                fontsize='medium')

        self.writers[writer].add_figure(caption, fig)

    def save_image(self, writer, caption, classes, predictions, labelss, step=None):
        ncols, nrows = 2, len(predictions)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 20))

        for i, (prediction, labels) in enumerate(zip(predictions, labelss)):
            for j, (title, data) in enumerate(zip(['Predicted', 'Ground Truth'], [prediction, labels])):
                ax[i][j].imshow(data, cmap='CMRmap', vmin=0, vmax=len(classes) - 1)
                ax[i][j].set_title(title, fontsize=10, color='blue')
                ax[i][j].axis('off')

        fig.set_tight_layout(True)

        self.writers[writer].add_figure(caption, fig, step)

    @staticmethod
    def create_df(val_data, scores_data, classes, group_names, groups):

        new_groups = groups + [list(set(classes).difference(*groups))]
        new_group_names = group_names + ['Ungrouped']

        unique_counts = dict(zip(classes, np.zeros(len(classes))))
        for i, (_, volume, labels, _) in enumerate(val_data):
            _, counts = np.unique(labels, return_counts=True)
            for class_idx, count in enumerate(counts):
                unique_counts[classes[int(class_idx)]] += count

        classes_grouped = dict(
            zip(new_group_names, [sorted(group, key=lambda x: unique_counts[x]) for group in new_groups]))

        sorted_classes_flat = [item for sublist in list(classes_grouped.values()) for item in sublist]
        index = pd.MultiIndex.from_tuples(
            [(name, _class) for name in classes_grouped for _class in classes_grouped[name]], names=['Group', 'Class'])

        data_sorted = {vol_id: [x[1] for x in sorted(scores.items(), key=lambda x: sorted_classes_flat.index(x[0]))] for
                       vol_id, scores in scores_data.items()}

        return pd.DataFrame(data_sorted, index=index)

    def graph(self, model, x):
        self.writers['train'].add_graph(model, x)

    def close(self):
        for writer in self.writers.values():
            writer.close()

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


@dataclass
class Task:
    name: str
    measure: str
    measure_name: str
    size: Optional[int] = None
    short_name: Optional[str] = None


def get_overall_score(row: pd.Series, tasks: Sequence[Task], weighted: bool = False) -> float:
    overall_score = 0
    for task in tasks:
        if not weighted:
            overall_score += row[f'{task.name}_test_{task.measure}']
        else:
            overall_score += task.size * row[f'{task.name}_test_{task.measure}']

    if not weighted:
        return overall_score / len(tasks)
    else:
        total_size = sum([task.size for task in tasks])
        return overall_score / total_size


def visualize_boxplot(
    df: pd.DataFrame, tasks: Sequence[Task], order=None,
    cols: int = 2, row_height: int = 5, fig_width: int = 14,
    all_tasks: bool = True,
    savefig=False,
    filename='mtl_polish_roberta_vs_distilroberta_full_results_plot',
    add_legend=True,
    legend_labels: Optional[List[str]] = None,
    bbox_to_anchor=(1.0, 1.0), loc: int = 2, ncol: int = 1,
    only_fig_legend: bool = True,
    steps: bool = True, runtimes: bool = True, epochs: bool = False, **kwargs
):
    if all_tasks:
        rows = math.ceil((len(tasks) + 2) / cols)
    else:
        rows = math.ceil(2 / cols)

    if order is None:
        types = df['type'].unique()
        types = sorted(types, key=len)
    else:
        types = order

    x_ticks_labels = []
    for type in types:
        split_type_string = type.split('_')
        type_string = split_type_string[0]
        if len(split_type_string) > 1:
            type_string += '\n' + '\n'.join(split_type_string[1:])
        x_ticks_labels.append(type_string)

    fig, axis = plt.subplots(rows, cols, figsize=(fig_width, rows * row_height),
                             constrained_layout=True)
    if rows * cols == 1:
        axis = [axis]
    else:
        axis = axis.flatten()

    ax = axis[0]
    sns.boxplot(x='type', y=f'overall_score', order=types, ax=ax, data=df, **kwargs)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_xlabel(None)
    ax.set_ylabel('Score')
    ax.set_title('Overall')

    _manage_legend_for_ax(
        ax, add_legend, only_fig_legend, loc, ncol, bbox_to_anchor, legend_labels=legend_labels
    )

    ax = axis[1]
    sns.boxplot(x='type', y='weighted_overall_score', order=types, ax=ax, data=df, **kwargs)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_xlabel(None)
    ax.set_ylabel('Score')
    ax.set_title('Weighted overall')

    handles, labels, legend_title = _manage_legend_for_ax(
        ax, add_legend, only_fig_legend, loc, ncol, bbox_to_anchor, legend_labels=legend_labels
    )

    if all_tasks:
        for task, ax in zip(tasks, axis[2:]):
            title = task.name + ' - ' + human_format(task.size)

            sns.boxplot(x='type', y=f'{task.name}_test_{task.measure}', order=types, ax=ax, data=df,
                        **kwargs)
            ax.set_xticklabels(x_ticks_labels)
            ax.set_xlabel(None)
            ax.set_ylabel(task.measure_name)
            ax.set_title(title)

            handles, labels, legend_title = _manage_legend_for_ax(
                ax, add_legend, only_fig_legend, loc, ncol, bbox_to_anchor,
                legend_labels=legend_labels
            )

    if add_legend and only_fig_legend and len(handles) > 0:
        fig.legend(handles, labels, title=legend_title, bbox_to_anchor=bbox_to_anchor, loc=loc,
                   ncol=ncol)

    if all_tasks:
        for ax in axis[len(tasks) + 2:]:
            ax.axis('off')

    sns.despine(fig)

    if savefig:
        fig.savefig(f'{filename}.pdf', bbox_inches='tight')

    additional_plots_prop = []
    if runtimes:
        additional_plots_prop.append(('Runtime', 'Runtime [s]'))
    if steps:
        additional_plots_prop.append(('trainer/global_step', 'Steps'))
    if epochs:
        additional_plots_prop.append(('epoch', 'Epochs'))

    hue = kwargs['hue'] if 'hue' in kwargs else None
    hue_order = kwargs['hue_order'] if 'hue_order' in kwargs else None

    rows = math.ceil((len(tasks)) / cols)

    for column_name, ylabel in additional_plots_prop:
        fig, axis = plt.subplots(rows, cols, figsize=(fig_width, rows * row_height),
                                 constrained_layout=True)
        if rows * cols == 1:
            axis = [axis]
        else:
            axis = axis.flatten()

        for task, ax in zip(tasks, axis):
            displ_df = df.dropna(subset=['dataset'])
            displ_df = displ_df[displ_df['dataset'].str.contains(task.name)]
            sns.barplot(x='type', y=column_name, order=types, ax=ax, data=displ_df,
                        edgecolor='black', hue=hue, hue_order=hue_order)
            ax.set_xticklabels(x_ticks_labels)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(None)
            title = task.name + ' - ' + human_format(task.size)
            ax.set_title(title)

            handles, labels, legend_title = _manage_legend_for_ax(
                ax, add_legend, only_fig_legend, loc, ncol, bbox_to_anchor,
                legend_labels=legend_labels
            )

        if only_fig_legend and len(handles) > 0:
            fig.legend(handles, labels, title=legend_title, bbox_to_anchor=bbox_to_anchor, loc=loc,
                       ncol=ncol)

        for ax in axis[len(tasks):]:
            ax.axis('off')

        sns.despine(fig)


def plot_gain_matrix(
    df: pd.DataFrame, tasks: Sequence[Task], mtl_types: List[str], v_bound: float = 30,
    scale_class_loss: str = 'none',
    uncertainty_loss: bool = False,
    fig_height: float = 10, col_width: float = 2.7,
    savefig: bool = False, filename: str = 'MTL_vs_STL_gains'
):
    vmin = -1 * v_bound
    vmax = v_bound

    cols = len(mtl_types)

    constrained_layout = True if cols > 1 else False
    fig, axis = plt.subplots(1, cols, figsize=(col_width * cols, fig_height),
                             constrained_layout=constrained_layout)

    if cols == 1:
        axis = [axis]

    for i, (mtl_type, ax) in enumerate(zip(mtl_types, axis)):

        df_mat = df[
            (df['type'].isin(['STL', mtl_type])) &
            (df['scale_class_loss'] == scale_class_loss) &
            (df['uncertainty_loss'] == uncertainty_loss)
            ]

        g = df_mat.groupby(
            ['dataset', 'learning_kind', 'model_name', 'mt_dataset_type', 'model_type',
             'scale_class_loss', 'type', 'model_kind'],
            as_index=False, sort=False, dropna=False
        ).mean()

        records = []

        for task in tasks:
            name = task.name
            measure = f'{task.name}_test_{task.measure}'
            record = {'task name': name}

            for model in g['model_name'].unique():
                df_model = g[g['model_name'] == model]
                MTL_score = df_model[df_model['learning_kind'] == 'MTL'][measure].iloc[0]
                STL_score = df_model[
                    (df_model['learning_kind'] == 'STL') & (df_model['dataset'] == task.name)][
                    measure].iloc[0]

                gain = MTL_score - STL_score

                model = model.replace('bert', 'BERT')
                model = model[0].upper() + model[1:]
                record[model] = 100 * gain

            records.append(record)

        df_records = pd.DataFrame.from_records(records, index='task name')
        cmap = sns.diverging_palette(10, 133, s=80, sep=1, as_cmap=True)
        cbar_kws = {'shrink': 0.75, 'label': 'Measure gain'}

        yticklabels = 'auto' if i == 0 else False
        cbar = True if i == cols - 1 else False
        sns.heatmap(df_records, linewidth=3, vmin=vmin, vmax=vmax, annot=True, fmt='.1f',
                    yticklabels=yticklabels,
                    square=True, ax=ax, cbar=cbar, cmap=cmap, cbar_kws=cbar_kws)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False,
                       labeltop=True)
        ax.tick_params(axis='x', labelrotation=45)

        split_t_string = mtl_type.split('_')
        if len(split_t_string) > 1:
            t_string = '\n'.join(split_t_string[1:])
        else:
            t_string = split_t_string[0]

        if cols == 1:
            title = u'MTL vs STL \u2212 gains' + '\n' + t_string
            pad = 0
        else:
            title = t_string
            pad = 20
        ax.set_title(title, pad=pad)

    if cols > 1:
        _ = fig.suptitle(u'MTL vs STL \u2212 gains')

    if savefig:
        fig.savefig(f'{filename}.pdf', bbox_inches='tight')


def _manage_legend_for_ax(
    ax: plt.Axes, add_legend: bool, only_fig_legend: bool,
    loc: Union[str, int, Tuple[float, float]], ncol: int,
    bbox_to_anchor: Union[Tuple[float, float], Tuple[float, float, float, float]],
    legend_labels: List[str] = None,
):
    handles, labels = ax.get_legend_handles_labels()
    legend_title = ''

    if legend_labels is not None:
        labels = legend_labels

    if ax.get_legend():
        legend_title = ax.get_legend().get_title().get_text().replace('_', ' ').capitalize()
        ax.get_legend().remove()
        if add_legend and not only_fig_legend:
            ax.legend(handles, labels, title=legend_title, bbox_to_anchor=bbox_to_anchor,
                      loc=loc, ncol=ncol)
    return handles, labels, legend_title


def human_format(num: float) -> str:
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

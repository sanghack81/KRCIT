import warnings
from os import mkdir
from os.path import exists

import matplotlib

matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel
from joblib import delayed
from pyrcds.domain import RelationalSchema, EntityClass, RelationshipClass, Cardinality
from pyrcds.model import RelationalVariable, RelationalPath
from tqdm import trange

from uai2017experiments.new_algos import ci_test_all
from uai2017experiments.run_complex_experiments import generate_structure
from uai2017experiments.run_simple_experiments import generate_values
from uai2017experiments.utils import AUPC

A, B, C, D = es = [EntityClass('A', 'X'), EntityClass('B', 'Y'), EntityClass('C', 'U'), EntityClass('D', 'V')]
X, Y = next(iter(A.attrs)), next(iter(B.attrs))

AB, AC, BD = rs = [RelationshipClass('R_AB', [], {A: Cardinality.many, B: Cardinality.many}),
                   RelationshipClass('R_AC', [], {A: Cardinality.many, C: Cardinality.many}),
                   RelationshipClass('R_BD', [], {B: Cardinality.many, D: Cardinality.many})]

schema = RelationalSchema(es, rs)


def complex_random_test(seed, randomness, n, mu, sd, independent, vertex_kernel_hop, slope):
    np.random.seed(seed)
    # np random seed?
    U = RelationalVariable(RelationalPath([A, AB, B]), Y)
    V = RelationalVariable(RelationalPath([A]), X)
    W = None

    # 1. Structuring
    skeleton = generate_structure(n, randomness)

    # 2. Values
    if independent and slope != 0:
        warnings.warn('independent option with non-zero slope={} is detected.'.format(slope))
    if independent:
        slope = 0

    generate_values(independent, mu, sd, skeleton, slope)

    p_values = ci_test_all(skeleton, U, V, W, vertex_kernel_hop=vertex_kernel_hop, gamma_x=0.5 / sd ** 2,
                           gamma_y=0.5 / sd ** 2, gamma_z=0.5 / sd ** 2, use_median=False)
    return (seed, randomness, n, mu, sd, independent, vertex_kernel_hop, slope, *p_values)


def draw_complex_null():
    for postfix in ['', '_vk1']:
        columns = ['seed', 'randomness', 'n', 'mu', 'sd', 'independent', 'vertex_kernel_hop', 'slope', 'HSIC', 'dummy1',
                   'R-HSIC', 'dummy2', 'RK-HSIC', 'dummy3', 'KRCIT-K', 'KRCIT-SD']
        df_all = pd.read_csv('new_results/complex_null_hypothesis{}.csv'.format(postfix), names=columns)
        df_all = pd.melt(df_all,
                         id_vars=['seed', 'randomness', 'n', 'mu', 'sd', 'independent', 'vertex_kernel_hop', 'slope'],
                         value_vars=['HSIC', 'dummy1', 'R-HSIC', 'dummy2', 'RK-HSIC', 'dummy3', 'KRCIT-K', 'KRCIT-SD'],
                         var_name='method',
                         value_name='p-value')

        for method, df_2 in df_all.groupby(by=['method']):
            if method.startswith('dummy'):
                continue
            for to_be_unique in ['n', 'sd', 'independent', 'vertex_kernel_hop', 'slope']:
                assert len(df_2[to_be_unique].unique()) == 1

            matrix_val = np.zeros([11, 11])
            for (random_val, mu_val), y in df_2.groupby(['randomness', 'mu'])['p-value']:
                row = int(random_val * 10)
                col = int(mu_val * 10)
                assert 0. <= row <= 10.
                assert 0. <= col <= 10.
                matrix_val[row, col] += sum(y < 0.05)

            # small row, high y-value = biased
            # large col, large x-value = heterogeneity
            sns.set(font_scale=1.2)
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(3.5, 3.15)
            cbar_ax = fig.add_axes([0.91, .25, .02, .5])
            sns.heatmap(matrix_val, vmin=0, vmax=20, ax=ax, cbar=True, cbar_ax=cbar_ax, xticklabels=[], yticklabels=[])
            cbar_ax.set_yticklabels([])
            # ax.set(xlabel='non-randomness of relationship', ylabel='heterogeneity')
            ax.set(xlabel='heterogeneity', ylabel='non-randomness')
            ax.set_title(method)
            plt.savefig('new_figures/complex_null_{}{}.pdf'.format(method, postfix), transparent=True, bbox_inches='tight', pad_inches=0.02)
            plt.close()


def draw_complex_alternative():
    for postfix in ['', '_vk1']:
        columns = ['seed', 'randomness', 'n', 'mu', 'sd', 'independent', 'vertex_kernel_hop', 'slope', 'HSIC',
                   'dummy1', 'R-HSIC', 'dummy2', 'RK-HSIC', 'dummy3', 'KRCIT-K', 'KRCIT-SD']
        df_all = pd.read_csv('new_results/complex_alternative_hypothesis{}.csv'.format(postfix), names=columns)
        df_all = pd.melt(df_all,
                         id_vars=['seed', 'randomness', 'n', 'mu', 'sd', 'independent', 'vertex_kernel_hop', 'slope'],
                         value_vars=['HSIC', 'dummy1', 'R-HSIC', 'dummy2', 'RK-HSIC', 'dummy3', 'KRCIT-K', 'KRCIT-SD'],
                         var_name='method',
                         value_name='p-value')
        df_all = df_all.drop(df_all[df_all['method'].str.startswith('dummy')].index)

        df_all.n = df_all.n.astype('int64')
        df_all.mu = df_all.mu.astype('float64')

        left_condition = df_all.mu == 0.0
        right_condition = df_all.mu != 0.0

        df_left = pd.concat([df_all[left_condition]], ignore_index=True)
        df_right = pd.concat([df_all[right_condition]], ignore_index=True)

        df_left['position'] = 1
        df_right['position'] = 2

        all_df = pd.concat([df_left, df_right], ignore_index=True)

        del df_all['randomness']
        del df_all['seed']
        del df_all['n']
        del df_all['mu']
        del df_all['sd']
        del df_all['independent']
        del df_all['vertex_kernel_hop']

        #
        sns.set(style='white', font_scale=1.4)
        paper_rc = {'lines.linewidth': 0.5, 'lines.markersize': 2}
        sns.set_context("paper", rc=paper_rc)
        plt.gcf().set_size_inches(7, 3.15)

        pal = sns.color_palette("Paired", 8)
        pal = [pal[1], pal[3], pal[5], pal[6], pal[7]]

        all_df = all_df.groupby(by=['slope', 'method', 'position']).aggregate(AUPC).reset_index()

        g = sns.factorplot(x='slope', y='p-value', hue='method', col='position', ci=0, data=all_df, size=2.2,
                           aspect=1.2, palette=pal, scale=2,
                           hue_order=['HSIC', 'R-HSIC', 'RK-HSIC', 'KRCIT-K', 'KRCIT-SD'], legend=False)
        g.set_xticklabels(step=4)
        g.set_xlabels('linear coefficient')
        g.set_ylabels('AUPC')
        titles = ["Biased + Homogeneous", "Randomized + Heterogeneous"]
        for i, (ax, title) in enumerate(zip(g.axes.flat, titles)):
            ax.set_title('')
            ax.text(.5, .95, title, horizontalalignment='center', transform=ax.transAxes)
            ax.set_ylim(0.4, 1.1)
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
            else:
                ax.legend([], [])

        plt.savefig('new_figures/complex_alternative{}_AUPC.pdf'.format(postfix),
                    transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()


def main():
    if not exists('new_results'):
        mkdir('new_results')
    if not exists('new_figures'):
        mkdir('new_figures')

    np.random.seed(0)

    n = 400
    sd = 0.1
    for vertex_kernel_hop in [1, 2]:  # only 1 and 2
        if vertex_kernel_hop == 1:
            postfix = '_vk1'
        else:
            postfix = ''

        # # null hypothesis tests
        if not exists('new_results/complex_null_hypothesis{}.csv'.format(postfix)):
            print('running complex experiments, null hypothesis')
            null_configurations = []
            for mixup in np.linspace(0., 1., 10 + 1):
                for mu in np.linspace(0., 1., 10 + 1):
                    null_configurations.append([mixup, n, mu, sd, True, vertex_kernel_hop, 0.0])

            for seed in trange(20):
                outss = Parallel(6)(delayed(complex_random_test)(i * 20 + seed, *config) for i, config in enumerate(null_configurations))
                with open('new_results/complex_null_hypothesis{}.csv'.format(postfix), 'a') as f:
                    for outs in outss:
                        print(*outs, sep=',', file=f)

        # alternative hypothesis tests
        if not exists('new_results/complex_alternative_hypothesis{}.csv'.format(postfix)):
            print('running complex experiments, alternative hypothesis')
            alt_configurations = []
            for slope in np.linspace(0, 1, 20 + 1):
                alt_configurations.append((0., n, 0., sd, False, vertex_kernel_hop, slope))
                alt_configurations.append((1., n, 1., sd, False, vertex_kernel_hop, slope))

            for seed in trange(200):
                outss = Parallel(21)(delayed(complex_random_test)(i * 200 + seed, *config) for i, config in enumerate(alt_configurations))
                with open('new_results/complex_alternative_hypothesis{}.csv'.format(postfix), 'a') as f:
                    for outs in outss:
                        print(*outs, sep=',', file=f)

    draw_complex_null()
    draw_complex_alternative()


if __name__ == '__main__':
    draw_complex_null()
    draw_complex_alternative()
    # main()

import itertools
import multiprocessing
import time
from configparser import ConfigParser
from os.path import exists

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uai2017experiments.run_rcm_experiments_new import unfinished_tasks, dummy


def main():
    num_trials = 200
    expected_size_per_entity_class = [200, 400, 600, 800]  # data
    max_degree_per_relationship = [1, 2, 3]  # data
    graph_kernel_hops = [-1, 1, 2, 3, 4]  # test
    hypotheses = ['null', 'alternative']

    configs = set()
    configs |= set(itertools.product(expected_size_per_entity_class, [3], [1], hypotheses))
    configs |= set(itertools.product([800], max_degree_per_relationship, [1], hypotheses))
    configs |= set(itertools.product([800], [3], graph_kernel_hops, hypotheses))

    more_configs = set()
    more_configs |= set(
        itertools.product(expected_size_per_entity_class, max_degree_per_relationship, graph_kernel_hops, hypotheses))
    configs = more_configs - configs

    configs = sorted(configs)
    configs = [(i * num_trials + trial, *config) for trial in range(num_trials) for i, config in enumerate(configs)]

    filename = 'new_results/conditional_more.csv'
    if exists(filename):
        df = pd.read_csv(filename, names=['seed', 'n', 'max_rel', 'hops', 'hypothesis'], usecols=[0, 1, 2, 3, 4])
        configs = unfinished_tasks(configs, df, ['seed', 'n', 'max_rel', 'hops', 'hypothesis'])

    np.random.shuffle(configs)

    previous_batch_size = None
    previous_n_jobs = None
    start = time.time()
    total = len(configs)
    while configs:
        parser = ConfigParser()
        parser.read('run_rcm.ini')
        batch_size = int(parser['LINUX' if multiprocessing.cpu_count() == 32 else 'IMAC']['batch_size'])
        n_jobs = int(parser['LINUX' if multiprocessing.cpu_count() == 32 else 'IMAC']['n_jobs'])
        if previous_batch_size != batch_size:
            print('new batch size: {}'.format(batch_size))
        if previous_n_jobs != n_jobs:
            print('new num threads: {}'.format(n_jobs))
        previous_batch_size, previous_n_jobs = batch_size, n_jobs

        subconfigs = configs[:batch_size]
        configs = configs[batch_size:]
        outss = Parallel(n_jobs)(delayed(dummy)(*config) for config in subconfigs)
        with open(filename, 'a') as f:
            for outs in outss:
                print(*outs, sep=',', file=f)

        passed = time.time() - start
        remain = len(configs)
        avg_time_per_config = passed / (total - remain)
        remain_secs = avg_time_per_config * remain
        hours = remain_secs // 3600
        minutes = (remain_secs % 3600) // 60
        secs = remain_secs % 60
        print('{}h {}m {}s left'.format(int(hours), int(minutes), int(secs)))


if __name__ == '__main__':
    main()

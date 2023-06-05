from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import lib

PROGRESS_BAR = False
OUT_DIR = "./out"


def run_1st_exp():
  n_runs = 2
  cmd = f'--num-applicants 1000 --applicants-per-round 100 --clip --methods ols 2sls --stream'
  args = lib.get_args(cmd)
  args_list = [(s, args) for s in np.arange(n_runs)]
  with Pool(n_runs) as p:
    runs = p.starmap(lib.run_multi_exp,
                     tqdm(args_list, total=len(args_list), disable=not PROGRESS_BAR))

  df = lib.runs2df(runs)
  dflong = pd.melt(df, id_vars='iterations', value_vars=('ols_env0', '2sls_env0'),
                   var_name='method',
                   value_name='error')
  _, ax = plt.subplots()
  sns.lineplot(data=dflong, x='iterations', y='error', hue='method', ax=ax)
  ax.set_ylim(bottom=-0.001, top=0.2)
  ax.set_ylabel(r'$|| \theta^* - \hat{hat}|| $')
  ax.set_xlabel('number of rounds')
  ax.grid()

  plt.savefig(f"{OUT_DIR}/01.pdf", format="pdf")
  return


def run_2nd_exp():
  n_runs = 2
  cmd = f'--num-applicants 1000 --applicants-per-round 100 --fixed-effort-conversion --scaled-duplicates sequence --b-bias 2 --num-envs 1 --pref uniform --methods ours 2sls ols'
  args = lib.get_args(cmd)
  args_list = [(s, args) for s in np.arange(n_runs)]
  with Pool(n_runs) as p:
    runs = p.starmap(lib.run_multi_exp,
                     tqdm(args_list, total=len(args_list), disable=not PROGRESS_BAR))
  df = lib.runs2df(runs)

  # long format for plotting
  value_vars = [f'{m}_env{e}' for m in args.methods for e in range(args.num_envs)]
  dflong = pd.melt(df, id_vars='iterations', value_vars=value_vars, var_name='env',
                   value_name='error')
  dflong['method'] = dflong.env.apply(lambda x: x.split('_')[0])
  dflong['env'] = dflong.env.apply(lambda x: x.split('_')[-1])

  # dflong = pd.melt(df, id_vars='iterations', value_vars=('ours_env0', 'ours_env1'), var_name='env', value_name='error')
  _, ax = plt.subplots()
  sns.lineplot(dflong, x='iterations', y='error', errorbar=('ci', 95), ax=ax, hue='method')
  ax.grid()
  ax.set_ylim(bottom=-0.001, top=.2)
  ax.set_ylabel(r'$|| \theta^* - \hat{hat}|| $')
  ax.set_xlabel('number of rounds')

  plt.savefig(f"{OUT_DIR}/02.pdf", format="pdf")
  return


def run_3rd_exp():
  n_runs = 4
  cmd = f'--num-applicants 5000 --applicants-per-round 100 --fixed-effort-conversion --scaled-duplicates sequence --b-bias 2 --num-envs 2 --pref uniform --methods ours 2sls ols'
  args = lib.get_args(cmd)
  args_list = [(s, args) for s in np.arange(n_runs)]
  with Pool(n_runs) as p:
    runs = p.starmap(lib.run_multi_exp,
                     tqdm(args_list, total=len(args_list), disable=not PROGRESS_BAR))
  df = lib.runs2df(runs)

  value_vars = [f'{m}_env{e}' for m in args.methods for e in range(args.num_envs)]
  dflong = pd.melt(df, id_vars='iterations', value_vars=value_vars, var_name='env',
                   value_name='error')
  dflong['method'] = dflong.env.apply(lambda x: x.split('_')[0])
  dflong['env'] = dflong.env.apply(lambda x: x.split('_')[-1])

  fig, ax = plt.subplots()
  sns.lineplot(data=dflong, x='iterations', y='error', hue='method', style='env',
               errorbar=('ci', 95), ax=ax)
  ax.grid()
  ax.set_ylim(bottom=-.001, top=.5)
  ax.set_ylabel(r'$ || \theta^* - \hat{\theta}|| $')
  ax.set_xlabel('number of rounds')

  plt.savefig(f"{OUT_DIR}/03.pdf", format="pdf")
  return


if __name__ == "__main__":
  run_1st_exp()
  run_2nd_exp()
  # run_3rd_exp()

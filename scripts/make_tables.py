#!/usr/bin/env python3
"""
Main results table (full-data point estimates, best value per column in bold),
a table with the 95% CI of every value, and the fit-convergence table.

Point estimates use the same correlation function as the bootstrap
(common.correlations), so every value lies inside its interval (checked).
The real-world dataset is reported with SRCC and KRCC only: with 32 videos in
8 scenes, the nonlinear fit behind PLCC and RMSE is not reliable.

Writes to <out_dir>: table_main.tex, table_ci.tex, table_fit_convergence.tex,
point_estimates.csv

Usage: python make_tables.py <explicit_csv> <martin_csv> <bootstrap_csv> <out_dir>
"""

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, METRICS, N_BOOTSTRAP, correlations

explicit_csv, martin_csv, boot_csv, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)

data = {'explicit': pd.read_csv(explicit_csv), 'martin': pd.read_csv(martin_csv)}
pe, fit_ok = {}, {}
for key, df in data.items():
    rows = {}
    for col, (name, hib, _) in METRICS.items():
        r = correlations(df, col, hib)
        rows[name] = {k.upper(): r[k] for k in ('srcc', 'krcc', 'plcc', 'rmse')}
        fit_ok[(key, name)] = r['fit_ok']
    pe[key] = pd.DataFrame(rows).T

bs = pd.read_csv(boot_csv).set_index(['Dataset', 'Metric'])
ci = lambda ds, m, k: (bs.loc[(ds, m), f'{k}_ci_low'], bs.loc[(ds, m), f'{k}_ci_high'])

layout = [('explicit', ['SRCC', 'KRCC', 'PLCC', 'RMSE']), ('martin', ['SRCC', 'KRCC'])]
best = {(ds, k): (pe[ds][k].round(3).idxmin() if k == 'RMSE' else pe[ds][k].round(3).idxmax())
        for ds, keys in layout for k in keys}
order = pe['explicit']['SRCC'].sort_values(ascending=False).index

header = [r"\begin{tabular}{l|cccc|cc}", r"\hline",
          r"\multirow{2}{*}{\textbf{Metric}} & \multicolumn{4}{c|}{\textbf{%s ($n = %d$)}} & "
          r"\multicolumn{2}{c}{\textbf{%s ($n = %d$)}} \\" % (DATASETS['explicit'], len(data['explicit']),
                                                                DATASETS['martin'], len(data['martin'])),
          r"\cline{2-7}",
          r" & \textbf{SRCC} & \textbf{KRCC} & \textbf{PLCC} & \textbf{RMSE} & \textbf{SRCC} & \textbf{KRCC} \\",
          r"\hline"]
main, supp = list(header), list(header)
for m in order:
    cells = []
    for ds, keys in layout:
        for k in keys:
            s = f"{pe[ds].loc[m, k]:.3f}"
            cells.append(f"\\textbf{{{s}}}" if best[(ds, k)] == m else s)
    main.append(f"{m} & " + " & ".join(cells) + r" \\")
    supp.append(f"{m} & " + " & ".join(f"[{ci(ds, m, k)[0]:.3f}, {ci(ds, m, k)[1]:.3f}]"
                                       for ds, keys in layout for k in keys) + r" \\")
footer = [r"\hline", r"\end{tabular}"]
open(os.path.join(out_dir, 'table_main.tex'), 'w').write("\n".join(main + footer) + "\n")
open(os.path.join(out_dir, 'table_ci.tex'), 'w').write("\n".join(supp + footer) + "\n")

# fit convergence: full data and share of bootstrap resamples with a failed logistic fit
conv = [r"\begin{tabular}{l|cc|cc}", r"\hline",
        r"\multirow{2}{*}{\textbf{Metric}} & \multicolumn{2}{c|}{\textbf{%s}} & \multicolumn{2}{c}{\textbf{%s}} \\"
        % (DATASETS['explicit'], DATASETS['martin']),
        r"\cline{2-5}",
        r" & \textbf{Full data} & \textbf{Resamples failed} & \textbf{Full data} & \textbf{Resamples failed} \\",
        r"\hline"]
for m in order:
    cells = []
    for ds in ('explicit', 'martin'):
        cells += ["converged" if fit_ok[(ds, m)] else "failed",
                  f"{100 * bs.loc[(ds, m), 'fit_failures'] / N_BOOTSTRAP:.1f}\\%"]
    conv.append(f"{m} & " + " & ".join(cells) + r" \\")
open(os.path.join(out_dir, 'table_fit_convergence.tex'), 'w').write("\n".join(conv + footer) + "\n")

bad = [(ds, m, k) for ds, keys in layout for k in keys for m in order
       if not (ci(ds, m, k)[0] - 1e-9 <= pe[ds].loc[m, k] <= ci(ds, m, k)[1] + 1e-9)]
print("point estimates outside their CI:", bad or "none")
pd.concat(pe, names=['Dataset', 'Metric']).to_csv(os.path.join(out_dir, 'point_estimates.csv'))

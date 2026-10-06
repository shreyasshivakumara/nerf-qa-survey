"""
Settings and correlation measures shared by the analysis scripts.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, linregress, pearsonr, spearmanr

N_BOOTSTRAP = 2000
SEED = 42

# key -> label used in the tables
DATASETS = {'explicit': 'Explicit-NeRF-QA', 'martin': 'Martin et al.'}

# column in the merged CSV -> (display name, higher is better,
#                              category: full-reference, no-reference or NeRF-specific)
# Lower-is-better metrics are negated before computing correlations.
METRICS = {
    'psnr': ('PSNR', True, 'FR'),
    'ssim': ('SSIM', True, 'FR'),
    'ms_ssim': ('MS-SSIM', True, 'FR'),
    'cw_ssim': ('CW-SSIM', True, 'FR'),
    'fsim': ('FSIM', True, 'FR'),
    'vif': ('VIF', True, 'FR'),
    'lpips': ('LPIPS', False, 'FR'),
    'lpips-vgg': ('LPIPS-VGG', False, 'FR'),
    'lpips+': ('LPIPS+', False, 'FR'),
    'dists': ('DISTS', False, 'FR'),
    'flip': ('FLIP', False, 'FR'),
    'vmaf': ('VMAF', True, 'FR'),
    'fovvideovdp': ('FovVideoVDP', True, 'FR'),
    'strred': ('STRRED', False, 'FR'),
    'niqe': ('NIQE', False, 'NR'),
    'piqe': ('PIQE', False, 'NR'),
    'maniqa': ('MANIQA', True, 'NR'),
    'musiq': ('MUSIQ', True, 'NR'),
    'cgvqm_2': ('CGVQM-2', True, 'NeRF'),
    'cgvqm_5': ('CGVQM-5', True, 'NeRF'),
    'nvs_sqa': ('NVS-SQA', True, 'NeRF'),
}
NAME_TO_COL = {name: col for col, (name, _, _) in METRICS.items()}


def logistic_4pl(x, b1, b2, b3, b4):
    """4-parameter logistic mapping from metric scores to MOS."""
    return b1 + (b2 - b1) / (1 + np.exp(-b4 * (x - b3)))


def correlations(df, col, higher_is_better, mos_col='MOS'):
    """
    SRCC and KRCC on the raw scores; PLCC and RMSE after a 4-parameter
    logistic fit (linear fit if the logistic fit does not converge).

    Returns a dict with srcc, krcc, plcc, rmse and fit_ok (False when the
    logistic fit failed and the linear fallback was used, or when the fitted
    curve is flat so that PLCC is undefined).
    """
    nan = dict(srcc=np.nan, krcc=np.nan, plcc=np.nan, rmse=np.nan, fit_ok=False)
    valid = df[col].notna() & df[mos_col].notna()
    if valid.sum() < 4:
        return nan
    x = df.loc[valid, col].values
    y = df.loc[valid, mos_col].values
    if not higher_is_better:
        x = -x
    try:
        srcc = spearmanr(x, y)[0]
        krcc = kendalltau(x, y)[0]
        try:
            p0 = [np.min(y), np.max(y), np.median(x), 1.0]
            params, _ = curve_fit(logistic_4pl, x, y, p0=p0, maxfev=10000)
            pred, fit_ok = logistic_4pl(x, *params), True
        except Exception:
            slope, intercept = linregress(x, y)[:2]
            pred, fit_ok = slope * x + intercept, False
        fit_ok = fit_ok and np.ptp(pred) > 0
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        plcc = pearsonr(pred, y)[0]
        return dict(srcc=srcc, krcc=krcc, plcc=plcc, rmse=rmse, fit_ok=fit_ok)
    except Exception:
        return nan

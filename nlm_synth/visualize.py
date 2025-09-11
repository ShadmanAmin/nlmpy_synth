import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_by_scale(df: pd.DataFrame, metric: str = 'morans_I', by: str='label'):
    fig, ax = plt.subplots(figsize=(6,4))
    for key, sub in df.groupby(by):
        g = sub.groupby('factor')[metric].mean().reset_index()
        ax.plot(g['factor'], g[metric], marker='o', label=str(key))
    ax.set_xlabel('Coarsening factor (block size)')
    ax.set_ylabel(metric.replace('_',' '))
    ax.set_title(f'{metric} vs. scale')
    ax.grid(True, ls=':', lw=0.6)
    ax.legend()
    plt.tight_layout()
    return fig, ax

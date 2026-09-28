import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

try:
    import scienceplots  # Registers 'science' and related matplotlib styles.
except ImportError:
    scienceplots = None

try:
    plt.style.use(['science', 'ieee'])
except OSError:
    # Fallback to default style so training/testing can proceed without SciencePlots styles.
    plt.style.use('default')
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'Times']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

os.makedirs('plots', exist_ok=True)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def smooth(y, box_pts=1):
    y = _to_numpy(y)
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plotter(name, y_true, y_pred, ascore, labels):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ascore = _to_numpy(ascore)
    labels = _to_numpy(labels)
    if 'TranAD' in name:
        y_true = np.roll(y_true, 1, axis=0)
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/output.pdf')
    point_labels = labels.reshape(-1) if labels.ndim == 1 or labels.shape[1] == 1 else None
    for dim in range(y_true.shape[1]):
        y_t, y_p, a_s = y_true[:, dim], y_pred[:, dim], ascore[:, dim]
        l = point_labels if point_labels is not None else labels[:, dim]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        # if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
        ax1.plot(smooth(y_t), linewidth=0.2, label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
        ax3 = ax1.twinx()
        ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
        if dim == 0:
            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        ax2.plot(smooth(a_s), linewidth=0.2, color='g')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        pdf.savefig(fig)
        plt.close()
    pdf.close()

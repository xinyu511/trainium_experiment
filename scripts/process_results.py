import pickle 
import matplotlib.pyplot as plt 
import numpy as np

snr_vals = [-5, -2.5, 0, 2.5, 5, 10]
stats = []

for snr in snr_vals:
    with open(f'cifar100_results_{snr}.pkl', 'rb') as f:
        data = pickle.load(f)

    data_np = np.array(data)
    stat = {
        "SNR": snr,
        "Mean": np.mean(data_np),
        "Max": np.max(data_np),
        "Min": np.min(data_np),
        "Std Dev": np.std(data_np),
    }
    stats.append(stat)

# Pretty print with tabulate if available
try:
    from tabulate import tabulate
    print(tabulate(stats, headers="keys", floatfmt=".4f"))
except ImportError:
    # Fallback print
    print(f"{'SNR':>6} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std Dev':>10}")
    for s in stats:
        print(f"{s['SNR']:>6} {s['Mean']:>10.4f} {s['Max']:>10.4f} {s['Min']:>10.4f} {s['Std Dev']:>10.4f}")

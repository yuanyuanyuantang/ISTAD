from src.parser import *
from src.folderconstants import *

dataset_alias = {
                'psm': 'PSM',
                'smd': 'SMD',
                'swat': 'SWaT',
                'exathlon': 'Exathlon',
        }
dataset_name = dataset_alias.get(args.dataset.lower(), args.dataset)

# Threshold parameters
lm_d = {
                'SMD': [(0.988, 1.00), (0.997, 1.06)],
                # 'SMD': [(0.997, 1.00), (0.997, 1.06)],
                'synthetic': [(0.999, 1), (0.999, 1)],
                'SWaT': [(0.9999, 1.2), (0.9999, 1.28)],
                'UCR': [(0.993, 1), (0.99935, 1)],
                'NAB': [(0.991, 1), (0.99, 1)],
                'SMAP': [(0.98, 1), (0.98, 1)],
                'MSL': [(0.97, 1), (0.999, 1.04)],
                'WADI': [(0.99, 1), (0.999, 1)],
                'MSDS': [(0.91, 1), (0.9, 1.04)],
                'MBA': [(0.87, 1), (0.93, 1.04)],
                'PSM': [(0.98, 0.9), (0.98, 0.9)],
                'Exathlon': [(0.99, 1), (0.99, 1)],
        }
lm = lm_d.get(dataset_name, [(0.99, 1), (0.99, 1)])[1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
                'SMD': 0.0001,
                'synthetic': 0.0001,
                'SWaT': 0.001,
                'SMAP': 0.001,
                'MSL': 0.002,
                'WADI': 0.0001,
                'MSDS': 0.001,
                'UCR': 0.006,
                'NAB': 0.009,
                'MBA': 0.001,
                'PSM': 0.0001,
                'Exathlon': 0.0001,
        }
lr = lr_d.get(dataset_name, 0.0001)

# Debugging
percentiles = {
                'SMD': (98, 2000),
                'synthetic': (95, 10),
                'SWaT': (95, 10),
                'SMAP': (97, 5000),
                'MSL': (97, 150),
                'WADI': (99, 1200),
                'MSDS': (96, 30),
                'UCR': (98, 2),
                'NAB': (98, 2),
                'MBA': (99, 2),
                'PSM': (98, 2000),
                'Exathlon': (98, 2000),
        }
percentile_merlin = percentiles.get(dataset_name, (98, 2000))[0]
cvp = percentiles.get(dataset_name, (98, 2000))[1]
preds = []
debug = 9

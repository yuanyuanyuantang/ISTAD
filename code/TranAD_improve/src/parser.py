import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                                        metavar='-d',
                                        type=str,
                                        required=False,
                                        default='synthetic',
                    help="dataset from ['synthetic', 'SMD', 'PSM', 'SWAT', 'Exathlon']")
parser.add_argument('--model',
                                        metavar='-m',
                                        type=str,
                                        required=False,
                                        default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--data_root',
                                        type=str,
                                        required=False,
                                        default=None,
                    help="root folder for dataset_tranad files; defaults to <project_root>/dataset_tranad")
parser.add_argument('--check_data',
                                        action='store_true',
                                        help="only run static dataset checks and exit")
parser.add_argument('--seed',
                                        type=int,
                                        required=False,
                                        default=42,
                    help="random seed for Python, NumPy, PyTorch and CUDA")
parser.add_argument('--test',
                                        action='store_true',
                                        help="test the model")
parser.add_argument('--retrain',
                                        action='store_true',
                                        help="retrain the model")
parser.add_argument('--less',
                                        action='store_true',
                                        help="train using less data")
args = parser.parse_args()

# import os

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import pandas as pd
# import math

# plt.switch_backend('agg')


# def adjust_learning_rate(optimizer, epoch, args):
#     # lr = args.learning_rate * (0.2 ** (epoch // 2))
#     if args.lradj == 'type1':
#         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
#     elif args.lradj == 'type2':
#         lr_adjust = {
#             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
#             10: 5e-7, 15: 1e-7, 20: 5e-8
#         }
#     elif args.lradj == 'type3':
#         lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
#     elif args.lradj == "cosine":
#         lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))


# class EarlyStopping:
#     def __init__(self, patience=7, verbose=False, delta=0):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.inf
#         self.delta = delta

#     def __call__(self, val_loss, model, path):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, path):
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
#         self.val_loss_min = val_loss


# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


# class StandardScaler():
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def transform(self, data):
#         return (data - self.mean) / self.std

#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean


# def visual(true, preds=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     if preds is not None:
#         plt.plot(preds, label='Prediction', linewidth=2)
#     plt.plot(true, label='GroundTruth', linewidth=2)
#     plt.legend()
#     plt.savefig(name, bbox_inches='tight')


# def adjustment(gt, pred):
#     anomaly_state = False
#     for i in range(len(gt)):
#         if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
#             anomaly_state = True
#             for j in range(i, -1, -1):
#                 if gt[j] == 0:
#                     break
#                 else:
#                     if pred[j] == 0:
#                         pred[j] = 1
#             for j in range(i, len(gt)):
#                 if gt[j] == 0:
#                     break
#                 else:
#                     if pred[j] == 0:
#                         pred[j] = 1
#         elif gt[i] == 0:
#             anomaly_state = False
#         if anomaly_state:
#             pred[i] = 1
#     return gt, pred


# def cal_accuracy(y_pred, y_true):
#     return np.mean(y_pred == y_true)
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def calc_point2point(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)

    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    return f1, precision, recall, TP, TN, FP, FN


def bf_search(
    score,
    label,
    start,
    end=None,
    step_num=1,
    display_freq=1,
    verbose=True,
    use_adjustment=True
):
    if verbose:
        print(f"\n{'=' * 60}")
        print("Start Best F1 threshold search...")
        print(f"{'=' * 60}")

    if step_num is None or end is None:
        end = start
        step_num = 1

    search_step, search_range, search_lower_bound = step_num, end - start, start

    if verbose:
        print(f"Search range: [{search_lower_bound:.4f}, {search_lower_bound + search_range:.4f}]")
        print(f"Search steps: {search_step}")
        print(f"Use point adjustment: {use_adjustment}")

    threshold = search_lower_bound
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_threshold = 0.0
    best_tp, best_tn, best_fp, best_fn = 0, 0, 0, 0
    best_latency = 0

    for i in range(search_step):
        threshold += search_range / float(search_step)

        if use_adjustment:
            predict = (score > threshold).astype(int)
            gt_adjusted, pred_adjusted = adjustment(label.copy(), predict.copy())
            target = calc_point2point(pred_adjusted, gt_adjusted)
            latency = 0
        else:
            predict = (score > threshold).astype(int)
            target = calc_point2point(predict, label)
            latency = 0

        f1, precision, recall, tp, tn, fp, fn = target

        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn
            best_latency = latency

        if verbose and i % display_freq == 0:
            print(f"Threshold: {threshold:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Best threshold: {best_threshold:.6f}")
        print(f"Best F1: {best_f1:.4f}")
        print(f"Precision: {best_precision:.4f}")
        print(f"Recall: {best_recall:.4f}")
        print(f"{'=' * 60}\n")

    return {
        "f1": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "TP": int(best_tp),
        "TN": int(best_tn),
        "FP": int(best_fp),
        "FN": int(best_fn),
        "threshold": float(best_threshold),
        "latency": float(best_latency),
        "use_adjustment": use_adjustment,
    }


def bf_search_adaptive(
    score,
    label,
    coarse_step_num=50,
    fine_step_num=100,
    verbose=True,
    use_adjustment=True
):
    score = np.asarray(score).reshape(-1)
    label = np.asarray(label).reshape(-1)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Adaptive Best F1 threshold search (two-stage)")
        print(f"{'=' * 60}")

    score_min = float(np.min(score))
    score_max = float(np.max(score))
    score_mean = float(np.mean(score))
    score_std = float(np.std(score))
    score_median = float(np.median(score))
    p1 = float(np.percentile(score, 1))
    p99 = float(np.percentile(score, 99))

    search_start = max(score_min, p1 * 0.5)
    search_end = min(score_max, p99 * 2.0)

    if verbose:
        print("\nScore statistics:")
        print(f"  Min: {score_min:.6f}")
        print(f"  Max: {score_max:.6f}")
        print(f"  Mean: {score_mean:.6f}")
        print(f"  Std: {score_std:.6f}")
        print(f"  Median: {score_median:.6f}")
        print(f"  P1: {p1:.6f}")
        print(f"  P99: {p99:.6f}")
        print(f"\nAuto search range: [{search_start:.6f}, {search_end:.6f}]")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Stage 1: coarse search (steps: {coarse_step_num})")
        print(f"{'=' * 60}")

    coarse_results = bf_search(
        score=score,
        label=label,
        start=search_start,
        end=search_end,
        step_num=coarse_step_num,
        display_freq=max(1, coarse_step_num // 10),
        verbose=verbose,
        use_adjustment=use_adjustment
    )

    coarse_best_threshold = coarse_results['threshold']
    search_range = search_end - search_start
    fine_range = search_range * 0.1
    fine_start = max(search_start, coarse_best_threshold - fine_range)
    fine_end = min(search_end, coarse_best_threshold + fine_range)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Stage 2: fine search (steps: {fine_step_num})")
        print(f"{'=' * 60}")
        print(f"Fine search around best threshold {coarse_best_threshold:.6f}")
        print(f"Fine search range: [{fine_start:.6f}, {fine_end:.6f}]")

    fine_results = bf_search(
        score=score,
        label=label,
        start=fine_start,
        end=fine_end,
        step_num=fine_step_num,
        display_freq=max(1, fine_step_num // 10),
        verbose=verbose,
        use_adjustment=use_adjustment
    )

    if fine_results['f1'] > coarse_results['f1']:
        final_results = fine_results
        if verbose:
            print("\nFine search found a better threshold.")
    else:
        final_results = coarse_results
        if verbose:
            print("\nCoarse search result remains best.")

    final_results['search_method'] = 'adaptive_two_stage'
    final_results['coarse_step_num'] = coarse_step_num
    final_results['fine_step_num'] = fine_step_num
    final_results['auto_search_range'] = [float(search_start), float(search_end)]
    final_results['score_statistics'] = {
        'min': float(score_min),
        'max': float(score_max),
        'mean': float(score_mean),
        'std': float(score_std),
        'median': float(score_median),
        'p1': float(p1),
        'p99': float(p99),
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("Adaptive search complete.")
        print(f"{'=' * 60}")
        print(f"Final best threshold: {final_results['threshold']:.6f}")
        print(f"Best F1: {final_results['f1']:.4f}")
        print(f"Precision: {final_results['precision']:.4f}")
        print(f"Recall: {final_results['recall']:.4f}")
        print(f"{'=' * 60}\n")

    return final_results


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

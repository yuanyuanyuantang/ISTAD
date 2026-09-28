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


def adjustment(gt, pred, entity_ids=None):
    """Apply point adjustment without joining independent entity timelines."""
    if entity_ids is not None:
        entities = np.asarray(entity_ids).reshape(-1)
        if len(entities) != len(gt):
            raise ValueError(
                f"entity_ids has {len(entities)} points, expected {len(gt)}"
            )
        gt_out = np.asarray(gt).copy()
        pred_out = np.asarray(pred).copy()
        for entity_id in np.unique(entities):
            keep = entities == entity_id
            entity_gt, entity_pred = adjustment(gt_out[keep], pred_out[keep])
            gt_out[keep] = entity_gt
            pred_out[keep] = entity_pred
        return gt_out, pred_out

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


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def calc_point2point(predict, actual):
    """
    计算点对点的评估指标
    
    参数:
        predict: 预测标签
        actual: 实际标签
        
    返回:
        (f1, precision, recall, TP, TN, FP, FN)
    """
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
    use_adjustment=True,
    entity_ids=None,
):
    """
    通过搜索最佳阈值来找到最佳 F1 分数（暴力搜索）
    
    参数:
        score: 异常分数数组
        label: 真实标签数组
        start: 搜索起始值
        end: 搜索结束值
        step_num: 搜索步数
        display_freq: 显示频率
        verbose: 是否打印详细信息
        use_adjustment: 是否使用点调整（Point Adjustment）
        
    返回:
        result_dict: 包含最佳 F1 和对应阈值的字典
    """
    if verbose:
        print(f"\n{'='*60}")
        print("开始 Best F1 阈值搜索...")
        print(f"{'='*60}")
    
    if step_num is None or end is None:
        end = start
        step_num = 1
    
    search_step, search_range, search_lower_bound = step_num, end - start, start
    
    if verbose:
        print(f"搜索范围: [{search_lower_bound:.4f}, {search_lower_bound + search_range:.4f}]")
        print(f"搜索步数: {search_step}")
        print(f"使用点调整: {use_adjustment}")
    
    threshold = search_lower_bound
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_threshold = 0.0
    best_tp, best_tn, best_fp, best_fn = 0, 0, 0, 0
    best_latency = 0

    # Point adjustment has a compact event-level equivalent: an anomaly segment
    # contributes its full length iff its maximum score crosses the threshold;
    # false positives are simply threshold crossings on normal points.  Compute
    # these sufficient statistics once instead of copying and scanning the full
    # timeline for every threshold candidate.
    if use_adjustment:
        score = np.asarray(score).reshape(-1)
        label = np.asarray(label).reshape(-1)
        actual = label.astype(bool)
        if entity_ids is None:
            entity_change = np.zeros(max(0, len(actual) - 1), dtype=bool)
        else:
            entities = np.asarray(entity_ids).reshape(-1)
            if len(entities) != len(actual):
                raise ValueError(
                    f"entity_ids has {len(entities)} points, expected {len(actual)}"
                )
            entity_change = entities[1:] != entities[:-1]
        starts = np.flatnonzero(
            actual & np.r_[True, (~actual[:-1]) | entity_change]
        )
        ends = np.flatnonzero(
            actual & np.r_[(~actual[1:]) | entity_change, True]
        ) + 1
        event_peak = np.asarray([
            np.max(score[start:end]) for start, end in zip(starts, ends)
        ])
        event_length = ends - starts
        normal_score = score[~actual]
        total_positive = int(actual.sum())
        total_negative = int((~actual).sum())
    
    for i in range(search_step):
        threshold += search_range / float(search_step)
        
        # 使用当前阈值进行预测
        if use_adjustment:
            hit_event = event_peak > threshold
            tp = int(event_length[hit_event].sum())
            fp = int(np.sum(normal_score > threshold))
            fn = total_positive - tp
            tn = total_negative - fp
            precision = tp / (tp + fp + 0.00001)
            recall = tp / (tp + fn + 0.00001)
            f1 = 2 * precision * recall / (precision + recall + 0.00001)
            target = f1, precision, recall, tp, tn, fp, fn
            latency = 0  # 简化版本，不计算延迟
        else:
            predict = (score > threshold).astype(int)
            target = calc_point2point(predict, label)
            latency = 0
        
        f1, precision, recall, tp, tn, fp, fn = target
        
        # 更新最佳结果
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn
            best_latency = latency
        
        if verbose and i % display_freq == 0:
            print(f"阈值: {threshold:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ 找到最佳阈值: {best_threshold:.6f}")
        print(f"  最佳 F1: {best_f1:.4f}")
        print(f"  Precision: {best_precision:.4f}")
        print(f"  Recall: {best_recall:.4f}")
        print(f"{'='*60}\n")
    
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
    use_adjustment=True,
    entity_ids=None,
):
    """
    自适应 Best F1 阈值搜索（两阶段：粗搜索 + 精细搜索）
    
    自动根据异常分数的分布确定搜索范围，无需手动设置参数。
    
    参数:
        score: 异常分数数组
        label: 真实标签数组
        coarse_step_num: 粗搜索步数（默认 50）
        fine_step_num: 精细搜索步数（默认 100）
        verbose: 是否打印详细信息
        use_adjustment: 是否使用点调整（Point Adjustment）
        
    返回:
        result_dict: 包含最佳 F1 和对应阈值的字典
    """
    score = np.asarray(score).reshape(-1)
    label = np.asarray(label).reshape(-1)
    
    if verbose:
        print(f"\n{'='*60}")
        print("自适应 Best F1 阈值搜索（两阶段）")
        print(f"{'='*60}")
    
    # 第一阶段：根据分数分布自动确定搜索范围
    score_min = float(np.min(score))
    score_max = float(np.max(score))
    score_mean = float(np.mean(score))
    score_std = float(np.std(score))
    score_median = float(np.median(score))
    
    # 计算合理的搜索范围
    # 使用百分位数来确定范围，避免极端值的影响
    p1 = float(np.percentile(score, 1))
    p99 = float(np.percentile(score, 99))
    
    # 搜索范围：从第1百分位到第99百分位的2倍
    search_start = max(score_min, p1 * 0.5)
    search_end = min(score_max, p99 * 2.0)
    
    if verbose:
        print(f"\n分数统计信息:")
        print(f"  最小值: {score_min:.6f}")
        print(f"  最大值: {score_max:.6f}")
        print(f"  均值: {score_mean:.6f}")
        print(f"  标准差: {score_std:.6f}")
        print(f"  中位数: {score_median:.6f}")
        print(f"  第1百分位: {p1:.6f}")
        print(f"  第99百分位: {p99:.6f}")
        print(f"\n自动确定的搜索范围: [{search_start:.6f}, {search_end:.6f}]")
    
    # 第一阶段：粗搜索
    if verbose:
        print(f"\n{'='*60}")
        print(f"阶段 1: 粗搜索（步数: {coarse_step_num}）")
        print(f"{'='*60}")
    
    coarse_results = bf_search(
        score=score,
        label=label,
        start=search_start,
        end=search_end,
        step_num=coarse_step_num,
        display_freq=max(1, coarse_step_num // 10),
        verbose=verbose,
        use_adjustment=use_adjustment,
        entity_ids=entity_ids,
    )
    
    coarse_best_threshold = coarse_results['threshold']
    coarse_best_f1 = coarse_results['f1']
    
    # 第二阶段：在最佳阈值附近进行精细搜索
    # 精细搜索范围：最佳阈值 ± 10% 的搜索范围
    search_range = search_end - search_start
    fine_range = search_range * 0.1
    fine_start = max(search_start, coarse_best_threshold - fine_range)
    fine_end = min(search_end, coarse_best_threshold + fine_range)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"阶段 2: 精细搜索（步数: {fine_step_num}）")
        print(f"{'='*60}")
        print(f"在最佳阈值 {coarse_best_threshold:.6f} 附近精细搜索")
        print(f"精细搜索范围: [{fine_start:.6f}, {fine_end:.6f}]")
    
    fine_results = bf_search(
        score=score,
        label=label,
        start=fine_start,
        end=fine_end,
        step_num=fine_step_num,
        display_freq=max(1, fine_step_num // 10),
        verbose=verbose,
        use_adjustment=use_adjustment,
        entity_ids=entity_ids,
    )
    
    # 比较两阶段的结果，选择更好的
    if fine_results['f1'] > coarse_results['f1']:
        final_results = fine_results
        if verbose:
            print(f"\n✓ 精细搜索找到更好的阈值！")
    else:
        final_results = coarse_results
        if verbose:
            print(f"\n✓ 粗搜索的结果已经是最优的。")
    
    # 添加搜索元信息
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
        print(f"\n{'='*60}")
        print(f"自适应搜索完成！")
        print(f"{'='*60}")
        print(f"最终最佳阈值: {final_results['threshold']:.6f}")
        print(f"最佳 F1: {final_results['f1']:.4f}")
        print(f"Precision: {final_results['precision']:.4f}")
        print(f"Recall: {final_results['recall']:.4f}")
        print(f"{'='*60}\n")
    
    return final_results

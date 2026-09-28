from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        epoch_times = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            epoch_cost = time.time() - epoch_time
            epoch_times.append(epoch_cost)
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_cost))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.avg_epoch_train_time = float(np.average(epoch_times))
        self.total_epoch_train_time = float(np.sum(epoch_times))
        self.finished_train_epochs = len(epoch_times)

        print("Average epoch training time: {:.4f}s".format(self.avg_epoch_train_time))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt_raw = test_labels.astype(int)
        gt = gt_raw.copy()

        # dump raw test scores/labels for offline metric computation (VUS etc.)
        if int(getattr(self.args, 'dump_scores_only', 0)):
            np.save(os.path.join(folder_path, 'test_scores.npy'), test_energy)
            np.save(os.path.join(folder_path, 'test_labels.npy'), gt_raw)
            print('Dumped test_scores.npy / test_labels.npy; skip protocol evaluation.')
            return

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        bestf1_results = None
        use_bestf1 = bool(int(getattr(self.args, 'use_bestf1_threshold', 0)))
        if use_bestf1:
            search_mode = str(getattr(self.args, 'bestf1_search_mode', 'adaptive')).lower()
            use_adjustment = bool(int(getattr(self.args, 'bestf1_use_adjustment', 1)))

            if search_mode == 'adaptive':
                from utils.tools import bf_search_adaptive

                coarse_step_num = int(getattr(self.args, 'bestf1_coarse_step_num', 50))
                fine_step_num = int(getattr(self.args, 'bestf1_fine_step_num', 100))

                print("\nUsing adaptive Best F1 search (two-stage)")
                print(f"Coarse steps: {coarse_step_num}, fine steps: {fine_step_num}")

                bestf1_results = bf_search_adaptive(
                    score=test_energy,
                    label=gt_raw,
                    coarse_step_num=coarse_step_num,
                    fine_step_num=fine_step_num,
                    verbose=True,
                    use_adjustment=use_adjustment
                )
            else:
                from utils.tools import bf_search

                search_start = float(getattr(self.args, 'bestf1_search_start', 0.01))
                search_end = float(getattr(self.args, 'bestf1_search_end', 2.0))
                search_step_num = int(getattr(self.args, 'bestf1_search_step_num', 100))

                print("\nUsing manual Best F1 search")
                print(f"Search range: [{search_start}, {search_end}], steps: {search_step_num}")

                bestf1_results = bf_search(
                    score=test_energy,
                    label=gt_raw,
                    start=search_start,
                    end=search_end,
                    step_num=search_step_num,
                    display_freq=max(1, search_step_num // 10),
                    verbose=True,
                    use_adjustment=use_adjustment
                )

            bestf1_results["roc_auc"] = float(roc_auc_score(gt_raw, test_energy))
            bestf1_results["pr_auc"] = float(average_precision_score(gt_raw, test_energy))

        result_path = os.path.join(folder_path, "result_anomaly_detection.txt")
        f = open(result_path, 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        if hasattr(self, "avg_epoch_train_time"):
            f.write("Average epoch training time: {:.4f}s\n".format(self.avg_epoch_train_time))
            f.write("Total epoch training time: {:.4f}s\n".format(self.total_epoch_train_time))
            f.write("Finished epochs: {}\n".format(self.finished_train_epochs))
        if bestf1_results is not None:
            f.write("\nBest F1 Threshold Method:\n")
            f.write("Threshold: {:.6f}\n".format(bestf1_results['threshold']))
            f.write("F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(
                bestf1_results['f1'], bestf1_results['precision'], bestf1_results['recall']))
            f.write("TP: {}, TN: {}, FP: {}, FN: {}\n".format(
                bestf1_results['TP'], bestf1_results['TN'],
                bestf1_results['FP'], bestf1_results['FN']))
            f.write("ROC-AUC: {:.4f}, PR-AUC: {:.4f}\n".format(
                bestf1_results['roc_auc'], bestf1_results['pr_auc']))
        f.write('\n')
        f.close()
        return

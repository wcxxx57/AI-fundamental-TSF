import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_Numeric_Long_Term(Exp_Basic):
    """Numeric-only long-term forecasting experiment.

    This class intentionally ignores any text/prior fields exposed by
    Dataset_Custom. It is the clean baseline path for PatchTST/DLinear.
    """

    def __init__(self, args):
        super(Exp_Numeric_Long_Term, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    def _select_criterion(self):
        return nn.MSELoss()

    def _output_root(self):
        output_dir = getattr(self.args, "output_dir", "")
        if output_dir:
            return Path(output_dir)
        return Path(__file__).resolve().parents[1] / "results" / "numeric_long_term"

    def _run_dir(self, setting):
        return self._output_root() / setting

    def _forward_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs, target

    @staticmethod
    def _inverse_if_possible(data_set, values):
        if not hasattr(data_set, "inverse_transform"):
            return values
        original_shape = values.shape
        try:
            return data_set.inverse_transform(values.reshape(-1, original_shape[-1])).reshape(original_shape)
        except Exception:
            return values

    @staticmethod
    def _append_history(path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _save_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, target = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(outputs.detach().cpu(), target.detach().cpu())
                total_loss.append(loss.item())

        self.model.train()
        return float(np.average(total_loss)) if total_loss else float("inf")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        run_dir = self._run_dir(setting)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(
            run_dir / "config.json",
            {
                "experiment": "numeric",
                "setting": setting,
                "args": vars(self.args),
                "run_dir": str(run_dir),
            },
        )

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, target = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(outputs, target)
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )

            train_loss_avg = float(np.average(train_loss)) if train_loss else float("inf")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            epoch_seconds = time.time() - epoch_time

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss_avg, vali_loss
                )
            )
            self._append_history(
                run_dir / "history.csv",
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss_avg,
                    "val_loss": vali_loss,
                    "epoch_seconds": epoch_seconds,
                    "iterations": iter_count,
                },
            )

            early_stopping(vali_loss, self.model, str(run_dir))
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = run_dir / "checkpoint.pth"
        if not best_model_path.exists():
            raise FileNotFoundError("Best checkpoint was not created: {}".format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        run_dir = self._run_dir(setting)

        if test:
            best_model_path = run_dir / "checkpoint.pth"
            if not best_model_path.exists():
                raise FileNotFoundError("Checkpoint not found for testing: {}".format(best_model_path))
            print("loading model")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, target = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(target.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0) if preds else np.empty((0, self.args.pred_len, self.args.c_out))
        trues = np.concatenate(trues, axis=0) if trues else np.empty((0, self.args.pred_len, self.args.c_out))
        print("test shape:", preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        preds_orig = self._inverse_if_possible(test_data, preds)
        trues_orig = self._inverse_if_possible(test_data, trues)
        mae_orig, mse_orig, rmse_orig, mape_orig, mspe_orig = metric(preds_orig, trues_orig)
        print("mse:{}, mae:{}".format(mse, mae))
        print("mse_orig:{}, mae_orig:{}".format(mse_orig, mae_orig))

        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "pred.npy", preds)
        np.save(run_dir / "true.npy", trues)
        np.save(run_dir / "pred_orig.npy", preds_orig)
        np.save(run_dir / "true_orig.npy", trues_orig)
        np.save(run_dir / "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))

        metrics = {
            "experiment": "numeric",
            "setting": setting,
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "mae_orig": float(mae_orig),
            "mse_orig": float(mse_orig),
            "rmse_orig": float(rmse_orig),
            "mape_orig": float(mape_orig),
            "mspe_orig": float(mspe_orig),
            "pred_shape": list(preds.shape),
            "true_shape": list(trues.shape),
            "run_dir": str(run_dir),
        }
        self._save_json(run_dir / "metrics.json", metrics)

        save_name = getattr(self.args, "save_name", "")
        if save_name:
            with open(save_name, "a", encoding="utf-8") as f:
                f.write(setting + "\n")
                f.write("mse:{}, mae:{}\n".format(mse, mae))
                f.write("mse_orig:{}, mae_orig:{}\n\n".format(mse_orig, mae_orig))

        return mse

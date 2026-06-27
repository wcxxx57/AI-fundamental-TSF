import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast, norm
from utils.metrics import metric
from utils.tools import EarlyStopping, visual


class VoTFrequencyDecompFusion(nn.Module):
    """VoT-style prediction-level frequency decomposition fusion.

    VoT decomposes numeric and text-side predictions into low, high, and middle
    frequency components, then learns six scalar weights over those components.
    This implementation keeps only that fusion idea and adapts it to the current
    fighting/ experiment interface. Inputs and outputs stay in [B, T, C].
    """

    component_names = [
        "numeric_low",
        "numeric_high",
        "numeric_mid",
        "text_low",
        "text_high",
        "text_mid",
    ]

    def __init__(self, init_prompt_weight=0.01, low_freq_ratio=0.1, high_freq_ratio=0.3):
        super().__init__()
        self.low_freq_ratio = float(low_freq_ratio)
        self.high_freq_ratio = float(high_freq_ratio)
        init_prompt_weight = float(init_prompt_weight)
        init_weights = torch.tensor(
            [
                1.0 - init_prompt_weight,
                1.0 - init_prompt_weight,
                1.0 - init_prompt_weight,
                init_prompt_weight,
                init_prompt_weight,
                init_prompt_weight,
            ],
            dtype=torch.float32,
        )
        self.decomp_w = nn.Parameter(init_weights)
        self.last_weights = None
        self.last_cutoffs = None

    def current_weights_tensor(self):
        return self.decomp_w

    def _effective_ratios(self, seq_len):
        freq_len = seq_len // 2 + 1
        if freq_len < 10:
            low_freq_ratio = min(0.3, 2.0 / max(freq_len, 1))
            high_freq_ratio = min(0.3, 2.0 / max(freq_len, 1))
        else:
            low_freq_ratio = self.low_freq_ratio
            high_freq_ratio = self.high_freq_ratio
        return low_freq_ratio, high_freq_ratio

    @staticmethod
    def _cutoffs(freq_len, low_freq_ratio, high_freq_ratio):
        low_cutoff = max(1, int(freq_len * low_freq_ratio))
        high_cutoff = min(freq_len - 1, int(freq_len * (1.0 - high_freq_ratio)))
        if low_cutoff >= high_cutoff:
            high_cutoff = min(freq_len, low_cutoff + 1)
        return low_cutoff, high_cutoff

    def frequency_decomposition(self, signal):
        """Split [B, C, T] into low, high, and middle frequency components."""
        seq_len = signal.shape[2]
        low_freq_ratio, high_freq_ratio = self._effective_ratios(seq_len)
        fft_signal = torch.fft.rfft(signal, dim=2)
        freq_len = fft_signal.shape[2]
        low_cutoff, high_cutoff = self._cutoffs(freq_len, low_freq_ratio, high_freq_ratio)
        self.last_cutoffs = {
            "seq_len": int(seq_len),
            "freq_len": int(freq_len),
            "low_cutoff": int(low_cutoff),
            "high_cutoff": int(high_cutoff),
            "low_freq_ratio": float(low_freq_ratio),
            "high_freq_ratio": float(high_freq_ratio),
        }

        low_fft = torch.zeros_like(fft_signal)
        high_fft = torch.zeros_like(fft_signal)
        mid_fft = torch.zeros_like(fft_signal)
        low_fft[:, :, :low_cutoff] = fft_signal[:, :, :low_cutoff]
        high_fft[:, :, high_cutoff:] = fft_signal[:, :, high_cutoff:]
        mid_fft[:, :, low_cutoff:high_cutoff] = fft_signal[:, :, low_cutoff:high_cutoff]

        low_freq = torch.fft.irfft(low_fft, n=seq_len, dim=2)
        high_freq = torch.fft.irfft(high_fft, n=seq_len, dim=2)
        mid_freq = torch.fft.irfft(mid_fft, n=seq_len, dim=2)
        return low_freq, high_freq, mid_freq

    def forward(self, num_pred, prompt_y):
        if num_pred.ndim != 3 or prompt_y.ndim != 3:
            raise ValueError("VoTFrequencyDecompFusion expects [B, T, C] tensors.")
        if num_pred.shape != prompt_y.shape:
            raise ValueError(
                "Numeric and text predictions must have the same shape, got {} and {}".format(
                    tuple(num_pred.shape),
                    tuple(prompt_y.shape),
                )
            )

        ts_pred = num_pred.transpose(1, 2)
        text_pred = prompt_y.transpose(1, 2)
        ts_low, ts_high, ts_mid = self.frequency_decomposition(ts_pred)
        text_low, text_high, text_mid = self.frequency_decomposition(text_pred)

        stacked_components = torch.stack(
            [ts_low, ts_high, ts_mid, text_low, text_high, text_mid],
            dim=0,
        )
        weights = self.decomp_w.view(6, 1, 1, 1)
        self.last_weights = self.decomp_w.detach().cpu()
        fused_signal = torch.sum(stacked_components * weights, dim=0)
        return fused_signal.transpose(1, 2)


class Exp_VoT_Frequency_Fusion_Forecast(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_VoT_Frequency_Fusion_Forecast, self).__init__(args)
        fusion_kwargs = {
            "init_prompt_weight": self.prompt_weight,
            "low_freq_ratio": getattr(args, "vot_low_freq_ratio", 0.1),
            "high_freq_ratio": getattr(args, "vot_high_freq_ratio", 0.3),
        }
        self.vot_frequency_fusion = VoTFrequencyDecompFusion(**fusion_kwargs).to(self.device)

    def _fusion_design(self):
        name = "vot_frequency_decomposition_fusion"
        initialization = "numeric components start at 1 - prompt_weight; text components start at prompt_weight"
        return {
            "name": name,
            "source_reference": "repo/VoT-main/exp/exp_long_term_forecasting_clip.py",
            "component_order": list(getattr(self.vot_frequency_fusion, "component_names", [])),
            "init_prompt_weight": self.prompt_weight,
            "low_freq_ratio": float(self.vot_frequency_fusion.low_freq_ratio),
            "high_freq_ratio": float(self.vot_frequency_fusion.high_freq_ratio),
            "initialization": initialization,
        }

    def _set_train_mode(self):
        self.model.train()
        self.mlp.train()
        self.mlp_proj.train()
        self.vot_frequency_fusion.train()

    def _set_eval_mode(self):
        self.model.eval()
        self.mlp.eval()
        self.mlp_proj.eval()
        self.vot_frequency_fusion.eval()

    def _select_optimizer_fusion(self):
        return optim.Adam(self.vot_frequency_fusion.parameters(), lr=self.args.learning_rate3)

    def _checkpoint_state(self, setting):
        state = super()._checkpoint_state(setting)
        state["vot_frequency_fusion"] = self.vot_frequency_fusion.state_dict()
        state["fusion_design"] = self._fusion_design()
        return state

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            if "mlp" in checkpoint:
                self.mlp.load_state_dict(checkpoint["mlp"])
            if "mlp_proj" in checkpoint:
                self.mlp_proj.load_state_dict(checkpoint["mlp_proj"])
            if "vot_frequency_fusion" in checkpoint:
                self.vot_frequency_fusion.load_state_dict(checkpoint["vot_frequency_fusion"])
        else:
            self.model.load_state_dict(checkpoint)

    def _encode_text(self, batch_text):
        if not self.Doc2Vec:
            prompt = [
                f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                for text_info in batch_text
            ]
            token_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(token_ids.to(self.device))
        else:
            prompt_embeddings = torch.tensor(
                [self.text_model.infer_vector(str(text).split()) for text in batch_text],
                dtype=torch.float32,
            ).to(self.device)

        if self.use_fullmodel:
            prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
        else:
            prompt_emb = prompt_embeddings
        return self.mlp(prompt_emb)

    def _pool_text(self, prompt_emb, num_pred):
        if self.Doc2Vec:
            return prompt_emb.unsqueeze(-1)

        if self.pool_type == "avg":
            pooled = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            return pooled.unsqueeze(-1)
        if self.pool_type == "max":
            pooled = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
            return pooled.unsqueeze(-1)
        if self.pool_type == "min":
            pooled = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
            return pooled.unsqueeze(-1)
        if self.pool_type == "attention":
            outputs_norm = F.normalize(num_pred, p=2, dim=1)
            prompt_norm = F.normalize(prompt_emb, p=2, dim=2)
            attention_scores = torch.bmm(prompt_norm, outputs_norm)
            attention_weights = F.softmax(attention_scores, dim=1)
            weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)
            return weighted_prompt_emb.unsqueeze(-1)

        raise ValueError("Unsupported pool_type: {}".format(self.pool_type))

    def _numeric_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        f_dim = -1 if self.args.features == "MS" else 0
        num_pred = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return num_pred, target

    def _forward_multimodal(self, data_set, batch_x, batch_y, batch_x_mark, batch_y_mark, index):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        prior_y = torch.from_numpy(data_set.get_prior_y(index)).float().to(self.device)
        if prior_y.ndim == 2:
            prior_y = prior_y.unsqueeze(-1)
        batch_text = data_set.get_text(index)

        prompt_emb = self._encode_text(batch_text)
        num_pred, target = self._numeric_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
        prompt_signal = self._pool_text(prompt_emb, num_pred)
        prompt_y = norm(prompt_signal) + prior_y
        outputs = self.vot_frequency_fusion(num_pred, prompt_y)
        return outputs, target

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self._set_eval_mode()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, index in vali_loader:
                outputs, target = self._forward_multimodal(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                )
                loss = criterion(outputs.detach().cpu(), target.detach().cpu())
                total_loss.append(loss.item())
        self._set_train_mode()
        return float(np.average(total_loss)) if total_loss else float("inf")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        run_dir = self._run_dir(setting)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(
            run_dir / "config.json",
            {
                "experiment": self._experiment_name(),
                "setting": setting,
                "fusion_design": self._fusion_design(),
                "args": vars(self.args),
                "run_dir": str(run_dir),
            },
        )

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_optim_mlp = self._select_optimizer_mlp()
        model_optim_proj = self._select_optimizer_proj()
        fusion_optim = self._select_optimizer_fusion()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self._set_train_mode()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                model_optim_mlp.zero_grad()
                model_optim_proj.zero_grad()
                fusion_optim.zero_grad()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, target = self._forward_multimodal(
                            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                        )
                        loss = criterion(outputs, target)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.step(model_optim_mlp)
                    scaler.step(model_optim_proj)
                    scaler.step(fusion_optim)
                    scaler.update()
                else:
                    outputs, target = self._forward_multimodal(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                    )
                    loss = criterion(outputs, target)
                    loss.backward()
                    model_optim.step()
                    model_optim_mlp.step()
                    model_optim_proj.step()
                    fusion_optim.step()

                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            train_loss = float(np.average(train_loss)) if train_loss else float("inf")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            self._append_history(
                run_dir / "history.csv",
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": vali_loss,
                    "epoch_seconds": time.time() - epoch_time,
                    "iterations": train_steps,
                },
            )

            previous_best = early_stopping.best_score
            early_stopping(vali_loss, self.model, str(run_dir))
            if early_stopping.best_score != previous_best:
                self._save_checkpoint(run_dir, setting)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = run_dir / "checkpoint.pth"
        if not best_model_path.exists():
            raise FileNotFoundError("Best checkpoint was not created: {}".format(best_model_path))
        self._load_checkpoint(best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        run_dir = self._run_dir(setting)
        if test:
            print("loading model")
            best_model_path = run_dir / "checkpoint.pth"
            if not best_model_path.exists():
                raise FileNotFoundError("Checkpoint not found for testing: {}".format(best_model_path))
            self._load_checkpoint(best_model_path)

        preds = []
        trues = []
        weight_history = []
        visual_dir = run_dir / "test_results"
        visual_dir.mkdir(parents=True, exist_ok=True)

        self._set_eval_mode()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                outputs, target = self._forward_multimodal(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                )
                if self.vot_frequency_fusion.last_weights is not None:
                    weight_history.append(self.vot_frequency_fusion.last_weights.numpy())

                pred = outputs.detach().cpu().numpy()
                true = target.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, str(visual_dir / "{}.pdf".format(i)))

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

        final_weights = self.vot_frequency_fusion.current_weights_tensor().detach().cpu().numpy()
        avg_test_weights = np.mean(np.stack(weight_history, axis=0), axis=0) if weight_history else final_weights
        component_names = list(getattr(self.vot_frequency_fusion, "component_names", []))
        metrics = {
            "experiment": self._experiment_name(),
            "setting": setting,
            "fusion_design": self._fusion_design(),
            "frequency_component_weights": {
                component_names[i]: float(final_weights[i]) for i in range(len(component_names))
            },
            "avg_test_frequency_component_weights": {
                component_names[i]: float(avg_test_weights[i]) for i in range(len(component_names))
            },
            "last_frequency_cutoffs": self.vot_frequency_fusion.last_cutoffs,
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
                f.write(setting + "  \n")
                f.write("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}\n".format(mse, mae, rmse, mape, mspe))
                f.write(
                    "mse_orig:{}, mae_orig:{}, rmse_orig:{}, mape_orig:{}, mspe:{}\n".format(
                        mse_orig, mae_orig, rmse_orig, mape_orig, mspe_orig
                    )
                )
                f.write("frequency_component_weights:{}\n\n".format(metrics["frequency_component_weights"]))

        return mse

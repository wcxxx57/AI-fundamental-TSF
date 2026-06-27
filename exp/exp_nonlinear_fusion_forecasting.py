import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from exp.conservative_fusion import ResidualShrinkageFusion
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast, norm
from utils.metrics import metric
from utils.tools import EarlyStopping, visual


class SimpleAdaptiveGateFusion(nn.Module):
    """Lightweight adaptive fusion for numeric and text/prior predictions.

    This is a constrained nonlinear extension of MM-TSFlib's fixed weighted sum.
    It learns one shared sigmoid gate from [Y_num, Y_text, Y_text - Y_num] with
    a tiny one-hidden-layer MLP and initializes exactly to the original
    prompt_weight.
    """

    def __init__(self, init_weight=0.01, hidden_dim=8):
        super().__init__()
        self.init_weight = float(init_weight)
        hidden_dim = max(2, int(hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.last_gate = None
        self._reset_to_linear_start()

    def _reset_to_linear_start(self):
        clipped = min(max(self.init_weight, 1e-4), 1.0 - 1e-4)
        gate_bias = float(np.log(clipped / (1.0 - clipped)))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, num_pred, prompt_y):
        features = torch.cat([num_pred, prompt_y, prompt_y - num_pred], dim=-1)
        gate = torch.sigmoid(self.gate(features))
        self.last_gate = gate.detach()
        return num_pred + gate * (prompt_y - num_pred)


class BoundedAdaptiveGateFusion(nn.Module):
    """Adaptive gate constrained to stay near the fixed MM-TSFlib weight."""

    def __init__(self, init_weight=0.01, hidden_dim=8, max_delta=0.03):
        super().__init__()
        self.init_weight = float(init_weight)
        self.max_delta = float(max_delta)
        if self.max_delta < 0.0:
            raise ValueError("max_delta must be non-negative.")
        hidden_dim = max(2, int(hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.last_gate = None
        self._reset_to_linear_start()

    def _reset_to_linear_start(self):
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, num_pred, prompt_y):
        features = torch.cat([num_pred, prompt_y, prompt_y - num_pred], dim=-1)
        raw_gate = self.gate(features)
        gate = self.init_weight + self.max_delta * torch.tanh(raw_gate)
        gate = torch.clamp(gate, 0.0, 1.0)
        self.last_gate = gate.detach()
        return num_pred + gate * (prompt_y - num_pred)


class Exp_Nonlinear_Fusion_Forecast(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Nonlinear_Fusion_Forecast, self).__init__(args)
        self.use_bounded_gate = self._experiment_name() == "mm_tsflib_nonlinear_bounded"
        self.use_residual_shrink = bool(getattr(args, "fusion_residual_shrink", 0)) or self._experiment_name().endswith(
            "_shrink"
        )
        if self.use_bounded_gate:
            self.nonlinear_fusion = BoundedAdaptiveGateFusion(
                init_weight=self.prompt_weight,
                hidden_dim=8,
                max_delta=getattr(args, "fusion_gate_delta_max", 0.03),
            ).to(self.device)
        else:
            self.nonlinear_fusion = SimpleAdaptiveGateFusion(init_weight=self.prompt_weight, hidden_dim=8).to(self.device)
        if self.use_residual_shrink:
            self.residual_shrinkage = ResidualShrinkageFusion(
                init_shrink=getattr(args, "fusion_shrink_init", 0.1),
                max_shrink=getattr(args, "fusion_shrink_max", 0.5),
                signed=bool(getattr(args, "fusion_shrink_signed", 0))
                or self._experiment_name().endswith("_shrink_signed"),
            ).to(self.device)
        else:
            self.residual_shrinkage = None
        self._gate_stats = []

    def _fusion_design(self):
        if self.use_bounded_gate:
            name = "bounded_mlp_adaptive_gate"
            gate_constraint = "gate = prompt_weight + max_delta * tanh(MLP(features))"
        else:
            name = (
                "simple_mlp_adaptive_gate_with_residual_shrinkage"
                if self.use_residual_shrink
                else "simple_mlp_adaptive_gate"
            )
            gate_constraint = "unbounded sigmoid gate in [0, 1]"
        return {
            "name": name,
            "init_weight": self.prompt_weight,
            "gate_hidden_dim": 8,
            "gate_constraint": gate_constraint,
            "gate_delta_max": float(getattr(self.nonlinear_fusion, "max_delta", 0.0)),
            "residual_shrinkage": self.use_residual_shrink,
            "shrink_init": float(getattr(self.args, "fusion_shrink_init", 0.1)),
            "shrink_max": float(getattr(self.args, "fusion_shrink_max", 0.5)),
            "shrink_signed": bool(getattr(self.args, "fusion_shrink_signed", 0))
            or self._experiment_name().endswith("_shrink_signed"),
        }

    def _set_train_mode(self):
        self.model.train()
        self.mlp.train()
        self.mlp_proj.train()
        self.nonlinear_fusion.train()
        if self.residual_shrinkage is not None:
            self.residual_shrinkage.train()

    def _set_eval_mode(self):
        self.model.eval()
        self.mlp.eval()
        self.mlp_proj.eval()
        self.nonlinear_fusion.eval()
        if self.residual_shrinkage is not None:
            self.residual_shrinkage.eval()

    def _select_optimizer_fusion(self):
        params = list(self.nonlinear_fusion.parameters())
        if self.residual_shrinkage is not None:
            params.extend(self.residual_shrinkage.parameters())
        return optim.Adam(params, lr=self.args.learning_rate3)

    def _checkpoint_state(self, setting):
        state = super()._checkpoint_state(setting)
        state["nonlinear_fusion"] = self.nonlinear_fusion.state_dict()
        if self.residual_shrinkage is not None:
            state["residual_shrinkage"] = self.residual_shrinkage.state_dict()
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
            if "nonlinear_fusion" in checkpoint:
                self.nonlinear_fusion.load_state_dict(checkpoint["nonlinear_fusion"])
            if self.residual_shrinkage is not None and "residual_shrinkage" in checkpoint:
                self.residual_shrinkage.load_state_dict(checkpoint["residual_shrinkage"])
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
        complex_outputs = self.nonlinear_fusion(num_pred, prompt_y)
        if self.residual_shrinkage is not None:
            base_outputs = (1.0 - self.prompt_weight) * num_pred + self.prompt_weight * prompt_y
            outputs = self.residual_shrinkage(base_outputs, complex_outputs)
        else:
            outputs = complex_outputs
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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

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
        gate_means = []
        visual_dir = run_dir / "test_results"
        visual_dir.mkdir(parents=True, exist_ok=True)

        self._set_eval_mode()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                outputs, target = self._forward_multimodal(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                )
                if self.nonlinear_fusion.last_gate is not None:
                    gate_means.append(float(self.nonlinear_fusion.last_gate.mean().detach().cpu()))

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

        metrics = {
            "experiment": self._experiment_name(),
            "setting": setting,
            "fusion_design": self._fusion_design(),
            "fusion_gate_hidden_dim": 8,
            "fusion_gate_mean": float(np.mean(gate_means)) if gate_means else None,
            "fusion_residual_shrinkage": bool(self.use_residual_shrink),
            "fusion_shrinkage": self.residual_shrinkage.value() if self.residual_shrinkage is not None else None,
            "fusion_shrinkage_max": float(getattr(self.args, "fusion_shrink_max", 0.5))
            if self.residual_shrinkage is not None
            else None,
            "fusion_shrinkage_signed": bool(getattr(self.args, "fusion_shrink_signed", 0))
            or self._experiment_name().endswith("_shrink_signed"),
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
                f.write("fusion_gate_mean:{}\n\n".format(metrics["fusion_gate_mean"]))

        return mse

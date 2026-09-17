import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from torchmetrics import (
    MeanSquaredError,
    SpearmanCorrCoef,
    PearsonCorrCoef,
    R2Score,
    MeanAbsoluteError,
)
from collections import OrderedDict

from gce_model import GatedCrossExpertModel


class LitModel(LightningModule):
    """
    对接当前 batch:

      - ec_ids:            [B', 4]
      - ec_mask:           [B', 4]
      - protein_embedding: [B, Lp, Dp]
      - protein_mask:      [B, Lp]
      - mol_embedding:     [B, Lm, Dm]
      - mol_mask:          [B, Lm]
      - macckeys:          [B, 167]
      - labels:            [B]

    训练目标:
        loss = alpha * loss_sup + (1 - alpha) * loss_avg

        loss_sup = 0.5 * [L(y1, y) + L(y2, y)]
        loss_avg = L(y_pred, y)

    base_loss:
        mse / huber / rmse
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.lr = args.lr

        self.model = GatedCrossExpertModel(args)

        self.base_loss = getattr(args, "base_loss", "rmse")
        # self.alpha = getattr(args, "alpha", 0.5)
        self.eps = getattr(args, "eps", 1e-7)

        if self.base_loss not in ["mse", "huber", "rmse"]:
            raise ValueError("Invalid base_loss, choose from ['mse', 'huber', 'rmse'].")

        self.valid_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("val_mse", MeanSquaredError()),
                    ("val_spearman_corr", SpearmanCorrCoef()),
                    ("val_pearson_corr", PearsonCorrCoef()),
                    ("val_r2", R2Score()),
                    ("val_mae", MeanAbsoluteError()),
                ]
            )
        )

        self.test_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("test_mse", MeanSquaredError()),
                    ("test_spearman_corr", SpearmanCorrCoef()),
                    ("test_pearson_corr", PearsonCorrCoef()),
                    ("test_r2", R2Score()),
                    ("test_mae", MeanAbsoluteError()),
                ]
            )
        )

    def configure_optimizers(self):
        return AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
        )

    def _call_model(self, batch):
        return self.model(
            protein_embedding=batch["protein_embedding"],
            protein_mask=batch["protein_mask"],
            mol_embedding=batch["mol_embedding"],
            mol_mask=batch["mol_mask"],
            ec_ids=batch["ec_ids"],
            ec_mask=batch["ec_mask"],
            ec_owner=batch["ec_owner"],
            macckeys=batch["macckeys"],
        )

    def _reg_loss(self, pred, labels):
        pred = pred.view(-1)
        labels = labels.view(-1)

        if self.base_loss == "mse":
            return F.mse_loss(pred, labels)

        if self.base_loss == "huber":
            return F.huber_loss(pred, labels)

        if self.base_loss == "rmse":
            return torch.sqrt(F.mse_loss(pred, labels) + self.eps)

        raise ValueError(f"Unsupported base_loss: {self.base_loss}")


    def _compute_fusion_loss(self, outputs, labels):
        labels = labels.float().view(-1)
        alpha = outputs["alpha"] # [B, 3]

        y_pred = outputs["pred"].view(-1)
        main_loss = self._reg_loss(y_pred, labels)
        gate_entropy = -(alpha * (alpha + self.eps).log()).sum(dim=-1).mean()
        loss_gate = -gate_entropy
        loss = main_loss + 0.01 * loss_gate # [B,] 


        loss_dict = {
            "total": loss.detach(),
        }

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        labels = batch["labels"].float().view(-1)
        bs = labels.shape[0]

        outputs = self._call_model(batch)

        loss, loss_dict = self._compute_fusion_loss(outputs, labels)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
        )

        for k, v in loss_dict.items():
            self.log(
                f"train_{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
            )

        return loss

    def _eval_step(self, batch, metrics, prefix):
        labels = batch["labels"].float().view(-1)
        bs = labels.shape[0]

        # eval/test 阶段使用 clean prediction
        pred = self._call_model(batch)
        pred = pred["pred"].view(-1)

        loss = self._reg_loss(pred, labels)
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
        )

        for name, metric in metrics.items():
            metric(pred, labels)
            self.log(
                name,
                metric,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=bs,
            )

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, self.valid_metrics, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, self.test_metrics, "test")
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .resnet1d import BasicBlock, Bottleneck, ResNet
from .unet import UNet
from .x_unet import XUnet


class UNetHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=(1, 1), padding=(0, 0), feature_name: str = "phase"
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_name = feature_name
        self.layers = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, features, targets=None, mask=None):

        x = features[self.feature_name]
        x = self.layers(x)

        loss = None
        if targets is not None:
            loss = self.losses(x, targets, mask)

        return x, loss

    def losses(self, inputs, targets, mask=None):
        """
        targets: (batch, channel, station, time)
        """
        inputs = inputs.float()
        log_targets = torch.nan_to_num(torch.log(targets))

        nx_in, nt_in = inputs.shape[-2:]
        nx_ta, nt_ta = targets.shape[-2:]
        assert nt_ta == nt_in
        if nx_ta != nx_in:
            inputs = F.interpolate(inputs, size=(nx_ta, nt_ta), mode="bilinear", align_corners=False)

        if mask is None:
            if self.out_channels == 1:
                min_loss = -(targets * log_targets + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
                loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") - min_loss
                loss = loss.mean()
                

                # inputs = torch.sigmoid(inputs)
                # loss = F.kl_div(inputs.log(), targets, reduction="none") + F.kl_div(
                #     (1 - inputs).log(), 1 - targets, reduction="none"
                # )
                # loss = torch.nan_to_num(loss)
                # loss = loss.mean()

            else:
                min_loss = -(targets * log_targets).sum(dim=1) # cross_entropy sum over dim=1
                loss = F.cross_entropy(inputs, targets, reduction="none") - min_loss
                loss = loss.mean()

                # inputs = torch.log_softmax(inputs, dim=1)
                # loss = F.kl_div(inputs, targets, reduction="none").sum(dim=1).mean()

                # focal loss
                # ce_loss = F.cross_entropy(inputs, targets, reduction="none")
                # pt = torch.exp(-ce_loss)
                # focal_loss = (1 - pt) ** 2 * ce_loss
                # loss = focal_loss.mean()
        else:
            mask = mask.type_as(inputs)
            mask_sum = mask.sum()
            if mask_sum == 0.0:
                mask_sum = 1.0

            
            if self.out_channels == 1:
                min_loss = -(targets * log_targets + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
                loss = (
                    torch.sum((F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") - min_loss) * mask)
                    / mask_sum
                )

                # inputs = torch.sigmoid(inputs)
                # kl_div = F.kl_div(inputs.log(), targets, reduction="none") + F.kl_div(
                #     (1 - inputs).log(), 1 - targets, reduction="none"
                # )
                # kl_div = torch.nan_to_num(kl_div)
                # loss = torch.sum(kl_div * mask) / mask_sum

            else:
                min_loss = -(targets * log_targets).sum(dim=1)
                loss = (
                    torch.sum((F.cross_entropy(inputs, targets, reduction="none") - min_loss) * mask.squeeze(1))
                    / mask_sum
                )  # cross_entropy already sum over dim=1

                # inputs = torch.log_softmax(inputs, dim=1)
                # loss = torch.sum(F.kl_div(inputs, targets, reduction="none").sum(dim=1) * mask.squeeze(1)) / mask_sum

                # focal loss
                # ce_loss = F.cross_entropy(inputs, targets, reduction="none")
                # pt = torch.exp(-ce_loss)
                # focal_loss = (1 - pt) ** 5 * ce_loss
                # loss = torch.sum(focal_loss * mask.squeeze(1)) / mask_sum

        return loss


class EventHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(1, 1),
        padding=(0, 0),
        scaling=1000.0,
        feature_name: str = "event",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_name = feature_name
        self.scaling = scaling
        # self.layers = nn.Conv2d(
        #     in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        # )
        # self.layers = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(7, 1), padding=(3, 0)),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        #     nn.LeakyReLU(),
        # )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
        )

    def forward(self, features, targets=None, mask=None):
        x = features[self.feature_name]
        x = self.layers(x) * self.scaling

        loss = None
        if targets is not None:
            loss = self.losses(x, targets, mask)
        return x, loss

    def losses(self, inputs, targets, mask=None):
        inputs = inputs.float()

        nx_in, nt_in = inputs.shape[-2:]
        nx_ta, nt_ta = targets.shape[-2:]
        assert nt_ta == nt_in
        if nx_ta != nx_in:
            inputs = F.interpolate(inputs, size=(nx_ta, nt_ta), mode="bilinear", align_corners=False)

        if mask is None:
            loss = F.mse_loss(inputs, targets) / self.scaling
        else:
            mask = mask.type_as(inputs)
            mask_sum = mask.sum()
            if mask_sum == 0.0:
                mask_sum = 1.0
            loss = torch.sum(F.l1_loss(inputs, targets, reduction="none") * mask) / mask_sum / self.scaling
            # loss = torch.sum(torch.abs(inputs - targets) * mask, dim=(1, 2, 3)).mean() / mask_sum

        return loss


class PhaseNet(nn.Module):
    def __init__(
        self,
        backbone="unet",
        log_scale=False,
        add_stft=False,
        add_polarity=False,
        add_event=False,
        add_prompt=False,
        event_center_loss_weight=1.0,
        event_time_loss_weight=1.0,
        polarity_loss_weight=1.0,
        prompt_loss_weight=1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_name = backbone
        self.add_stft = add_stft
        self.add_event = add_event
        self.add_polarity = add_polarity
        self.add_prompt = add_prompt
        self.event_center_loss_weight = event_center_loss_weight
        self.event_time_loss_weight = event_time_loss_weight
        self.polarity_loss_weight = polarity_loss_weight
        self.prompt_loss_weight = prompt_loss_weight

        if backbone == "unet":
            self.backbone = UNet(
                channels=3,
                dim=16,
                out_dim=32,
                log_scale=log_scale,
                add_stft=add_stft,
                add_polarity=add_polarity,
                add_event=add_event,
                add_prompt=add_prompt,
            )
        elif backbone == "xunet":
            self.backbone = XUnet(
                channels=3,
                dim=32,
                out_dim=64,
                log_scale=log_scale,
                add_stft=add_stft,
                add_polarity=add_polarity,
                add_event=add_event,
                add_prompt=add_prompt,
            )
        else:
            raise ValueError("backbone only supports unet or xunet")

        if backbone == "unet":
            self.phase_picker = UNetHead(32, 3, feature_name="phase")
            if self.add_polarity:
                self.polarity_picker = UNetHead(32, 1, feature_name="polarity")
            if self.add_event:
                self.event_detector = UNetHead(32, 1, feature_name="event")
                self.event_timer = EventHead(32, 1, feature_name="event")

        elif backbone == "xunet":
            self.phase_picker = UNetHead(64, 3, feature_name="phase")
            if self.add_polarity:
                self.polarity_picker = UNetHead(64, 1, feature_name="polarity")
            if self.add_event:
                self.event_detector = UNetHead(64, 1, feature_name="event")
                self.event_timer = EventHead(64, 1, feature_name="event")

        else:
            raise ValueError("backbone only supports unet or xunet")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:
        data = batched_inputs["data"].to(self.device)

        phase_pick = batched_inputs["phase_pick"].to(self.device) if "phase_pick" in batched_inputs else None
        phase_mask = batched_inputs["phase_mask"].to(self.device) if "phase_mask" in batched_inputs else None
        event_center = batched_inputs["event_center"].to(self.device) if "event_center" in batched_inputs else None
        event_time = batched_inputs["event_time"].to(self.device) if "event_time" in batched_inputs else None
        if "event_center_mask" in batched_inputs:
            event_center_mask = batched_inputs["event_center_mask"].to(self.device)
            event_time_mask = batched_inputs["event_time_mask"].to(self.device)
        else:
            event_center_mask = batched_inputs["event_mask"].to(self.device)
            event_time_mask = batched_inputs["event_mask"].to(self.device)
        polarity = batched_inputs["polarity"].to(self.device) if "polarity" in batched_inputs else None
        polarity_mask = batched_inputs["polarity_mask"].to(self.device) if "polarity_mask" in batched_inputs else None
        prompt_center = batched_inputs["prompt_center"].float() if "prompt_center" in batched_inputs else None
        if self.__class__.__name__ not in ["PhaseNetDAS"]:
            phase_mask = None
            event_center_mask = None
            
        if self.backbone_name == "swin2":
            station_location = batched_inputs["station_location"].to(self.device)
            features = self.backbone(data, station_location)
        else:
            features = self.backbone(data)

        output = {"loss": 0.0}
        output_phase, loss_phase = self.phase_picker(features, phase_pick, mask=phase_mask)
        output["phase"] = output_phase
        if loss_phase is not None:
            output["loss_phase"] = loss_phase
            output["loss"] += loss_phase

        if self.add_polarity:
            output_polarity, loss_polarity = self.polarity_picker(features, polarity, mask=polarity_mask)
            output["polarity"] = output_polarity
            if loss_polarity is not None:
                output["loss_polarity"] = loss_polarity * self.polarity_loss_weight
                output["loss"] += loss_polarity * self.polarity_loss_weight

        # if self.add_stft and self.training:
        if self.add_stft:
            output["spectrogram"] = features["spectrogram"]

        if self.add_event:
            output_event_center, loss_event_center = self.event_detector(features, event_center, mask=event_center_mask)
            output["event_center"] = output_event_center
            if loss_event_center is not None:
                output["loss_event_center"] = loss_event_center * self.event_center_loss_weight
                output["loss"] += loss_event_center * self.event_center_loss_weight
            output_event_time, loss_event_time = self.event_timer(features, event_time, mask=event_time_mask)
            output["event_time"] = output_event_time
            if loss_event_time is not None:
                output["loss_event_time"] = loss_event_time * self.event_time_loss_weight
                output["loss"] += loss_event_time * self.event_time_loss_weight

        return output


def build_model(
    backbone="unet",
    log_scale=True,
    *args,
    **kwargs,
) -> PhaseNet:
    return PhaseNet(
        backbone=backbone,
        log_scale=log_scale,
    )

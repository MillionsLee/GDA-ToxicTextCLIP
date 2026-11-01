import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from .evaluate import get_similarity_distance
import numpy as np


def get_loss(
    umodel,
    outputs,
    criterion,
    options,
    memory_bank,
    current_epoch,
    sample_index=None,
    sample_weight=None,
):
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    text_embeds_nn = None
    text_memory_bank = memory_bank

    if (options.distributed):
        gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
        gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

        dist.all_gather(gathered_image_embeds, image_embeds)
        dist.all_gather(gathered_text_embeds, text_embeds)

        image_embeds = torch.cat(
            gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
        text_embeds = torch.cat(
            gathered_text_embeds[:options.rank] + [text_embeds] + gathered_text_embeds[options.rank + 1:])

    if options.memory_bank and options.RoCLIP_switch and current_epoch % options.break_epoch == 0:
        text_embeds_nn = text_memory_bank(image_embeds, update=False)
        text_memory_bank(text_embeds, update=True)
        logits_text_per_image_zero = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = umodel.logit_scale.exp() * text_embeds_nn @ image_embeds.t()
        logits_text_per_image = logits_image_per_text.t()
        batch_size = len(logits_text_per_image)
    else:
        logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        logits_text_per_image_zero = logits_text_per_image.t()
        batch_size = len(logits_text_per_image)

    target = torch.arange(batch_size).long().to(options.device, non_blocking=True)

    # Per-sample symmetric CLIP contrastive loss.
    # When sample_weight is all ones, this is equivalent to the original loss.
    loss_i2t = F.cross_entropy(
        logits_text_per_image,
        target,
        reduction="none",
    )

    loss_t2i = F.cross_entropy(
        logits_image_per_text,
        target,
        reduction="none",
    )

    per_sample_loss = (loss_i2t + loss_t2i) / 2.0

    if sample_weight is None:
        crossmodal_contrastive_loss = per_sample_loss.mean()
    else:
        sample_weight = sample_weight.to(
            device=per_sample_loss.device,
            dtype=per_sample_loss.dtype,
        )

        crossmodal_contrastive_loss = (
            per_sample_loss * sample_weight
        ).sum() / sample_weight.sum()

    contrastive_loss = crossmodal_contrastive_loss
    loss = contrastive_loss

    return loss, contrastive_loss


def train(epoch, model, data, optimizer, scheduler, scaler, options, memory_bank):
    dataloader = data["train"]
    if (options.distributed): dataloader.sampler.set_epoch(epoch)
    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device)

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))
    umodel = model.module if (options.distributed) else model

    start = time.time()
    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")
    for index, batch in enumerate(dataloader):
        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        input_ids, attention_mask, pixel_values, sample_index = batch["input_ids"].to(options.device,
                                                                                      non_blocking=True), batch[
            "attention_mask"].to(options.device, non_blocking=True), batch["pixel_values"].to(options.device,
                                                                                              non_blocking=True), batch[
            "index"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        is_poison = batch.get("is_poison", None)

        sample_weight = None
        if is_poison is not None:
            is_poison = is_poison.to(
                options.device,
                non_blocking=True,
            )

            sample_weight = torch.ones(
                len(input_ids),
                device=options.device,
                dtype=torch.float32,
            )

            sample_weight[is_poison.bool()] = options.poison_weight

        with autocast():
            loss, contrastive_loss = get_loss(
                umodel,
                outputs,
                criterion,
                options,
                memory_bank,
                epoch,
                sample_index,
                sample_weight=sample_weight,
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)
        end = time.time()

        if (options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            logging.info(
                f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")
            metrics = {"loss": loss.item(), "contrastive_loss": contrastive_loss.item(), "time": end - start,
                       "lr": optimizer.param_groups[0]["lr"]}
            if (options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})

            start = time.time()
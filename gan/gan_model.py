import pytorch_lightning as pl
import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from pytorch_lightning.core.decorators import auto_move_data


from mlcore.config import config_add_options, ConfigEntry, str2bool
from mlcore import utils

from dense_model import DecoderWithAttention, DenseNetEncoder

from layers import BigGANGeneratorBlock, BigGANDiscriminatorBlock, SNSelfAttention2d

from models.gan.biggan import GANPix2Pix, GANDiscriminator

@config_add_options("gan")
def config_gan():
    return {
        "sync_bn": ConfigEntry(default=True, type=str2bool),
        "source_image": ConfigEntry(default="source_image"),
        "source_domain": ConfigEntry(default="source_domain"),
        "target_image": ConfigEntry(default="target_image"),
        "target_domain": ConfigEntry(default="target_domain"),
        "num_classes": ConfigEntry(default=2, type=int),
        "always_swap_domain": ConfigEntry(default=False, type=str2bool),
        "split_discriminator": ConfigEntry(default=False, type=str2bool),
        "z_dim": ConfigEntry(default=120, type=int),
        "use_self_attention":ConfigEntry(default=True, type=str2bool),
        "task_model": ConfigEntry(default="dense"),
        "task_model_weight": ConfigEntry(default=1.0, type=float),
        "task_model_weight_decay": ConfigEntry(default=0.0001, type=float),
    }


@config_add_options("dense")
def config_dense():
    return {
        "emb_dim": ConfigEntry(default=512),  # dimension of word embeddings
        "attention_dim": ConfigEntry(default=512),  # dimension of attention linear layers
        "decoder_dim": ConfigEntry(default=512),  # dimension of decoder RNN
        "dropout": ConfigEntry(default=0.5),
    }


@config_add_options("discriminator_optimizer")
def config_discriminator_optimizer():
    return {
        "lr": ConfigEntry(default=2e-5, type=float),  # learning rate for encoder if fine-tuning
        "beta_1": ConfigEntry(default=0.0, type=float),
        "beta_2": ConfigEntry(default=0.999, type=float),
        "eps": ConfigEntry(default=1e-8, type=float),
        "weight_decay": ConfigEntry(default=0.0, type=float),
    }


@config_add_options("generator_optimizer")
def config_generator_optimizer():
    return {
        "lr": ConfigEntry(default=5e-5, type=float),  # learning rate for encoder if fine-tuning
        "beta_1": ConfigEntry(default=0.0, type=float),
        "beta_2": ConfigEntry(default=0.999, type=float),
        "eps": ConfigEntry(default=1e-8, type=float),
        "weight_decay": ConfigEntry(default=0.0, type=float),
    }



class GANModel(pl.LightningModule):
    def __init__(self, params):
        super(GANModel, self).__init__()

        self.params = params

        self.generator = GANPix2Pix(use_self_attention=self.params.gan.use_self_attention)
        self.discriminator = GANDiscriminator(use_self_attention=self.params.gan.use_self_attention)

        if params.gan.task_model == "dense":

            self.recognition_decoder = DecoderWithAttention(
                attention_dim=self.params.dense.attention_dim,
                embed_dim=self.params.dense.emb_dim,
                decoder_dim=self.params.dense.decoder_dim,
                vocab_size=113,
                dropout=self.params.dense.dropout,
            )

            self.recognition_encoder = DenseNetEncoder()
        else:
            self.recognition_decoder = None
            self.recognition_encoder = None

        if params.gan.sync_bn:
            self.generator = nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
            self.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
            if self.recognition_decoder is not None:
                self.recognition_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.recognition_decoder)
            if self.recognition_encoder is not None:
                self.recognition_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.recognition_encoder)


    def configure_ddp(self, model, devices_ids):
        # SyncBacthNorm
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        return super(GANModel, self).configure_ddp(model, devices_ids)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        generator_optimizer = torch.optim.AdamW(
            params=self.generator.parameters(),
            lr=self.params.generator_optimizer.lr,
            betas=(self.params.generator_optimizer.beta_1, self.params.generator_optimizer.beta_2),
            eps=self.params.generator_optimizer.eps,
            weight_decay=self.params.generator_optimizer.weight_decay,
        )

        
        discriminator_parameters = [{'params':list(self.discriminator.parameters()), 'weight_decay': self.params.discriminator_optimizer.weight_decay}]


        task_model = {'params': [], 'weight_decay': self.params.gan.task_model_weight_decay}
        if self.recognition_encoder is not None:
            task_model['params'].extend(list(self.recognition_encoder.parameters()))

        if self.recognition_decoder is not None:
            task_model['params'].extend(list(self.recognition_decoder.parameters()))

        if len(task_model['params']) > 0:
            discriminator_parameters.append(task_model)

        discriminator_optimizer = torch.optim.AdamW(
            params=discriminator_parameters,
            lr=self.params.discriminator_optimizer.lr,
            betas=(self.params.discriminator_optimizer.beta_1, self.params.discriminator_optimizer.beta_2),
            eps=self.params.discriminator_optimizer.eps,
        )

        return [generator_optimizer, discriminator_optimizer], []

    # @auto_move_data
    def forward(self, source_image, z, domain):
        image = self.generator(source_image, z, domain)

        return image

    def training_step(self, batch, batch_idx, optimizer_idx):
        result = {}

        source_image = utils.get_element(batch, self.params.gan.source_image)
        source_domain = utils.get_element(batch, self.params.gan.source_domain)
        target_image = utils.get_element(batch, self.params.gan.target_image)
        target_domain = utils.get_element(batch, self.params.gan.target_domain)

        if not self.params.gan.split_discriminator:
            pad_target = torch.max(torch.as_tensor(source_image.shape[2:], dtype= torch.int)-torch.as_tensor(target_image.shape[2:], dtype= torch.int), torch.zeros(2, dtype= torch.int))
            pad_source = torch.max(-torch.as_tensor(source_image.shape[2:], dtype= torch.int)+torch.as_tensor(target_image.shape[2:], dtype= torch.int), torch.zeros(2, dtype= torch.int))

            target_image = F.pad(target_image, [0,pad_target[1], 0,pad_target[0]], "constant", 0)
            source_image = F.pad(source_image, [0,pad_source[1], 0,pad_source[0]], "constant", 0)


        # print(f"Target Domain: {target_domain}")

        target_domain = torch.LongTensor(target_domain).to(source_image.device)


        result.update({"source_image": source_image})
        result.update({"target_image": target_image})

        if optimizer_idx == 0:
            z = torch.randn([source_image.shape[0], self.params.gan.z_dim]).type_as(source_image)
            if self.params.gan.always_swap_domain:
                # TODO random choice should be better
                transfered_domain = (1-torch.LongTensor(source_domain)).to(z.device)
            else:
                transfered_domain = torch.randint(size=[source_image.shape[0]], high=self.params.gan.num_classes).to(z.device)

            self.source_image = source_image
            self.transfered_domain = transfered_domain

            self.transfered_image = self(source_image, z, transfered_domain)

            fake = self.discriminator(self.transfered_image, transfered_domain)

            g_loss = -torch.mean(fake)

            loss = [g_loss]

            if self.recognition_encoder is not None and self.recognition_decoder is not None:

                image_embedding = self.recognition_encoder(self.transfered_image)

                scores, caps_sorted, decode_lengths, alphas, sort_ind = self.recognition_decoder(
                    image_embedding,
                    batch["source_sequence"],
                    torch.sum(batch["source_sequence_mask"], dim=1),
                    device=self.transfered_image.device.index,
                )
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                recognition_loss = F.cross_entropy(scores, targets)

                loss.append(self.params.gan.task_model_weight * recognition_loss)
                result.update({"recognition_loss_transfered": recognition_loss})

            result.update({"transfered_image": self.transfered_image})
            result.update({"g_loss": g_loss})
            result.update({"loss": torch.sum(torch.stack(loss))})

        if optimizer_idx == 1:
            self.target_image = target_image
            
            if self.params.gan.split_discriminator:
                fake = self.discriminator(self.transfered_image.detach(), self.transfered_domain )
                # print("##############")
                # print(transfered_domain)
                # print(target_domain)
                real = self.discriminator(target_image, target_domain)
            else:
                dicriminator_output = self.discriminator(torch.cat([self.transfered_image.detach(),target_image], dim=0), torch.cat([self.transfered_domain,target_domain], dim =0))
                fake = dicriminator_output[:self.transfered_domain.shape[0]]
                real = dicriminator_output[self.transfered_domain.shape[0]:]

            d_loss_real = torch.mean(F.relu(1.0 - real))
            d_loss_fake = torch.mean(F.relu(1.0 + fake))

            loss = [d_loss_real, d_loss_fake]

            if self.recognition_encoder is not None and self.recognition_decoder is not None:

                image_embedding = self.recognition_encoder(self.target_image)

                scores, caps_sorted, decode_lengths, alphas, sort_ind = self.recognition_decoder(
                    image_embedding,
                    batch["target_sequence"],
                    torch.sum(batch["target_sequence_mask"], dim=1),
                    device=self.target_image.device.index,
                )
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                recognition_loss = F.cross_entropy(scores, targets)

                loss.append(self.params.gan.task_model_weight * recognition_loss)
                result.update({"recognition_loss_target": recognition_loss})

            result.update({"d_loss_real": d_loss_real})
            result.update({"d_loss_fake": d_loss_fake})
            result.update({"loss": torch.sum(torch.stack(loss))})

        return result

    def training_step_end(self, batch_parts_outputs):

        log = {"loss": batch_parts_outputs["loss"].mean()}
        progress_bar = {}

        if "d_loss_real" in batch_parts_outputs:
            log.update({"discriminator/loss/real": batch_parts_outputs["d_loss_real"]})
            progress_bar.update({"discriminator/loss": batch_parts_outputs["loss"].mean()})

        if "d_loss_fake" in batch_parts_outputs:
            log.update({"discriminator/loss/fake": batch_parts_outputs["d_loss_fake"]})

        if "g_loss" in batch_parts_outputs:
            log.update({"generator/loss": batch_parts_outputs["g_loss"]})
            progress_bar.update({"generator/loss": batch_parts_outputs["g_loss"].mean()})

        if "recognition_loss_target" in batch_parts_outputs:
            log.update({"discriminator/rec_loss_target": batch_parts_outputs["recognition_loss_target"]})
            progress_bar.update(
                {"discriminator/rec_loss_target": batch_parts_outputs["recognition_loss_target"].mean()}
            )

        if "recognition_loss_transfered" in batch_parts_outputs:
            log.update({"generator/rec_loss_transfered": batch_parts_outputs["recognition_loss_transfered"]})
            progress_bar.update(
                {"generator/rec_loss_transfered": batch_parts_outputs["recognition_loss_transfered"].mean()}
            )
        return {
            "loss": batch_parts_outputs["loss"].mean(),
            "progress_bar": progress_bar,
            "log": log,
        }


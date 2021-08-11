import torchvision
from torch import nn
from torch.nn import functional as F
import torch
from formula_model import FormulaOCR


from mlcore.config import config_add_options, ConfigEntry, str2bool
from torch.nn.utils.rnn import pack_padded_sequence

from torchvision.models.densenet import _DenseBlock, _Transition

from collections import OrderedDict

import os
import imageio

from formula_pipeline import read_dictionary

from pytorch_lightning.core.decorators import auto_move_data


class DenseNetEncoder(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(
        self,
        growth_rate=24,
        block_config=(16, 16, 16),
        num_init_features=48,
        bn_size=4,
        drop_rate=0.5,
        memory_efficient=False,
        grayscale=True,
        input_channel=1,
    ):

        super(DenseNetEncoder, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(input_channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return out


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=684, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, device=None):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]),
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class DenseModel(FormulaOCR):
    def __init__(self, *args, **kwargs):
        super(DenseModel, self).__init__(*args, **kwargs)

        self.decoder = DecoderWithAttention(
            attention_dim=self.params.model.attention_dim,
            embed_dim=self.params.model.emb_dim,
            decoder_dim=self.params.model.decoder_dim,
            vocab_size=113,
            dropout=self.params.model.dropout,
        )

        self.encoder = DenseNetEncoder()

        dictionary_path = kwargs.get("dictionary_path", None)

        if dictionary_path is not None:

            self.dictionary = read_dictionary(dictionary_path)
            self.inv_dictionary = {v: k for k, v in self.dictionary.items()}

    @auto_move_data
    def forward(self, image):

        return self.encoder(image)

    def training_step(self, batch, batch_idx):

        # REQUIRED

        image_embedding = self(batch["image"])

        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            image_embedding,
            batch["sequence"],
            torch.sum(batch["sequence_mask"], dim=1),
            device=batch["image"].device.index,
        )
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = F.cross_entropy(scores, targets)

        perplexity = torch.exp(loss)

        return {"loss": loss, "perplexity": perplexity, "image": batch["image"]}

    def training_step_end(self, batch_parts_outputs):

        if self.global_step % self.trainer.log_save_interval == 0:
            grid = torchvision.utils.make_grid(batch_parts_outputs["image"])
            self.logger.experiment.add_image(f"train/images", grid, self.global_step)
            self.logger.experiment.add_histogram(f"train/images", batch_parts_outputs["image"], self.global_step)

        return {
            "loss": batch_parts_outputs["loss"].mean(),
            "log": {
                "train_loss": batch_parts_outputs["loss"].mean(),
                "train_perplexity": batch_parts_outputs["perplexity"].mean(),
            },
            "progress_bar": {
                "train_loss": batch_parts_outputs["loss"].mean(),
                "train_perplexity": batch_parts_outputs["perplexity"].mean(),
            },
        }

    def validation_step(self, batch, batch_idx):
        # OPTIONAL

        image_embedding = self.encoder(batch["image"])
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            image_embedding,
            batch["sequence"].to(image_embedding.device),
            torch.sum(batch["sequence_mask"], dim=1).to(image_embedding.device),
            device=batch["image"].device.index,
        )
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        return {"val_loss": F.cross_entropy(scores, targets)}

    def validation_end(self, outputs):
        # print(outputs)
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        perplexity = torch.exp(avg_loss)

        # self.logger.experiment.add_scalar(f'val/loss', avg_loss, self.global_step)
        # self.logger.experiment.add_scalar(f'val/perplexity', perplexity, self.global_step)

        # global_step is for storing
        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "global_step": self.global_step,
            "log": {"val_loss": avg_loss, "val_perplexity": perplexity},
            "progress_bar": {"val_loss": avg_loss, "val_perplexity": perplexity},
        }

    def validation_epoch_end(self, outputs):
        # print(outputs)
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        perplexity = torch.exp(avg_loss)

        # self.logger.experiment.add_scalar(f'val/loss', avg_loss, self.global_step)
        # self.logger.experiment.add_scalar(f'val/perplexity', perplexity, self.global_step)

        # global_step is for storing
        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "global_step": self.global_step,
            "log": {"val_loss": avg_loss, "val_perplexity": perplexity},
        }

    # def test_step(self, batch, batch_idx):

    #     image_embedding = self(batch["image"])

    #     scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
    #         image_embedding,
    #         batch["sequence"],
    #         torch.sum(batch["sequence_mask"], dim=1),
    #         device=batch["image"].device.index,
    #     )
    #     targets = caps_sorted[:, 1:]
    #     scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    #     targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
    #     loss = F.cross_entropy(scores, targets)
    #     perplexity = torch.exp(loss)

    #     return {"test_loss": loss, "perplexity": perplexity, "image": batch["image"]}

    def test_step(self, batch, batch_idx):

        hypotheses = []

        k = 3  # self.params.test.beam_size

        image_embedding = self(batch["image"])
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            image_embedding,
            batch["sequence"].to(image_embedding.device),
            torch.sum(batch["sequence_mask"], dim=1).to(image_embedding.device),
            device=batch["image"].device.index,
        )

        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = F.cross_entropy(scores, targets)
        perplexity = torch.exp(loss).detach().cpu().numpy()

        loss = loss.detach().cpu().numpy()

        #
        enc_image_size = image_embedding.size(1)
        encoder_dim = image_embedding.size(3)

        # Flatten encoding
        image_embedding = image_embedding.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = image_embedding.size(1)

        # We'll treat the problem as having a batch size of k
        image_embedding = image_embedding.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.dictionary["<start>"]]] * k).cuda(batch["image"].device.index)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).cuda(batch["image"].device.index)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.decoder.init_hidden_state(image_embedding)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = self.decoder.attention(image_embedding, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            # print(scores)
            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // len(self.dictionary.keys())  # vocab_size  # (s)
            next_word_inds = top_k_words % len(self.dictionary.keys())  # vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds) if next_word != self.dictionary["<end>"]
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            image_embedding = image_embedding[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # print(top_k_scores)
            # Break if things have been going on too long
            if step > 500:
                break
            step += 1

        if len(complete_seqs_scores) == 0:
            seq = seqs[0].tolist()
        else:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

        # # References
        # img_caps = allcaps[0].tolist()
        # img_captions = list(
        #     map(lambda c: [w for w in c if w not in {self.dictionary['<start>'], self.dictionary['<end>'], self.dictionary['<pad>']}],
        #         img_caps))  # remove <start> and pads
        # references.append(img_captions)

        # Hypotheses

        result_idx = [
            w for w in seq if w not in {self.dictionary["<start>"], self.dictionary["<end>"], self.dictionary["<pad>"]}
        ]
        result_str = [self.inv_dictionary[w] for w in result_idx]

        gt = batch["sequence"].squeeze(0).cpu().numpy().tolist()

        gt_idx = [
            w for w in gt if w not in {self.dictionary["<start>"], self.dictionary["<end>"], self.dictionary["<pad>"]}
        ]
        gt_str = [self.inv_dictionary[w] for w in gt_idx]

        # print("########")
        # print(gt_str)
        # print(result_str)
        # if self.params.test.prediction_output_path is not None:
        #     image_out = os.path.join(self.params.test.prediction_output_path, "img")
        #     gt_out = os.path.join(self.params.test.prediction_output_path, "gt")
        #     res_out = os.path.join(self.params.test.prediction_output_path, "res")

        #     os.makedirs(image_out, exist_ok=True)
        #     os.makedirs(gt_out, exist_ok=True)
        #     os.makedirs(res_out, exist_ok=True)

        #     filename = os.path.splitext(os.path.basename(batch["path"][0]))[0]

        #     with open(os.path.join(gt_out, f"{filename}.tex"), "w") as f:
        #         f.write("$" + " ".join(gt_str) + "$\n")

        #     with open(os.path.join(res_out, f"{filename}.tex"), "w") as f:
        #         f.write("$" + " ".join(result_str) + "$\n")

        #     imageio.imwrite(
        #         os.path.join(image_out, f"{filename}.jpg"), batch["image"].squeeze(0).squeeze(0).cpu().numpy()
        #     )

        return {"loss": loss, "perplexity": perplexity, "gt_str": gt_str, "pred_str": result_str}

        # return {"test_loss": loss, "perplexity": perplexity, "image": batch["image"]}

    # def test_epoch_end(self, outputs):

    #     image_embedding = self(batch["image"])

    #     scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
    #         image_embedding,
    #         batch["sequence"],
    #         torch.sum(batch["sequence_mask"], dim=1),
    #         device=batch["image"].device.index,
    #     )
    #     targets = caps_sorted[:, 1:]
    #     scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    #     targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
    #     loss = F.cross_entropy(scores, targets)
    #     perplexity = torch.exp(loss)

    #     return {"loss": loss, "perplexity": perplexity, "image": batch["image"]}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.Adam(
            params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.params.optimizer.lr
        )

        return optimizer

    # def val_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(
    #         MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    #         batch_size=self.params.val_dataloader.batch_size,
    #     )

    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(
    #         MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    #         batch_size=self.params.test_dataloader.batch_size,
    #     )
    @config_add_options("optimizer")
    def config_optimizer():
        return {
            "lr": ConfigEntry(default=1e-4),  # learning rate for encoder if fine-tuning
            "grad_clip": ConfigEntry(default=5.0),
        }

    @config_add_options("model")
    def config_optimizer():
        return {
            "emb_dim": ConfigEntry(default=512),  # dimension of word embeddings
            "attention_dim": ConfigEntry(default=512),  # dimension of attention linear layers
            "decoder_dim": ConfigEntry(default=512),  # dimension of decoder RNN
            "dropout": ConfigEntry(default=0.5),
            "alpha_c": ConfigEntry(default=1.0),
        }


{
    # Training parameters
    # 'epochs': ConfigEntry(default=120),  # number of epochs to train for (if early stopping is not triggered)
    # 'epochs_since_improvement': ConfigEntry(default=0),  # keeps track of number of epochs since there's been an improvement in validation BLEU
    # # 'batch_size': ConfigEntry(default=32)
    # 'workers': ConfigEntry(default=1),  # for data-loading; right now, only 1 works with h5py
    # 'grad_clip': ConfigEntry(default=5.),  # clip gradients at an absolute value of
    # 'alpha_c': ConfigEntry(default=1.),  # regularization parameter for 'doubly stochastic attention', as in the paper
    # 'best_bleu4': ConfigEntry(default=0.),  # BLEU-4 score right now
    # 'print_freq': ConfigEntry(default=100),  # print training/validation stats every __ batches
    # 'fine_tune_encoder': ConfigEntry(default=False),  # fine-tune encoder?
    # 'checkpoint': ConfigEntry(default=None),  # path to checkpoint, None if none
}

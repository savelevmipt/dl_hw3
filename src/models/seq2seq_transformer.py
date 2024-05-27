import torch
import torch.nn as nn
import metrics

import numpy as np

from models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, device, emb_size, vocab_size, max_seq_len, target_tokenizer,
                 transformer_params, scheduler_params
                 ):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device

        self.emb_size = emb_size

        self.embedding = nn.Embedding(vocab_size, emb_size).to(device)
        self.pos_encoder = PositionalEncoding(emb_size, max_seq_len).to(device)

        self.model = nn.Transformer(d_model=emb_size, **transformer_params).to(device)
        self.decoder = nn.Linear(emb_size, vocab_size).to(device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)

        self.target_tokenizer = target_tokenizer

        self.src_mask = None
        self.trg_mask = None

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        src = self.pos_encoder(self.embedding(input_tensor).long() * np.sqrt(self.emb_size))
        trg = self.pos_encoder(self.embedding(target_tensor).long() * np.sqrt(self.emb_size))
        src, trg = src.permute(1, 0, 2), trg.permute(1, 0, 2)

        out = self.model(
            src,
            trg,
            self.model.generate_square_subsequent_mask(input_tensor.size(1), 
                                                       device=self.device),
            self.model.generate_square_subsequent_mask(input_tensor.size(1), 
                                                       device=self.device),
        )

        return torch.argmax(out, dim=-1).clone(), self.decoder(out).permute(1, 0, 2)

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch
        _, out = self.forward(input_tensor, target_tensor)
        target = target_tensor.reshape(-1)
        out = out.reshape(-1, out.shape[-1])
        loss = self.loss(out, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        _, out = self.forward(input_tensor, target_tensor)
        target = target_tensor.reshape(-1)
        out = out.reshape(-1, out.shape[-1])
        loss = self.loss(out, target)
        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = predicted.squeeze(-1).detach().cpu().numpy()[:, 1:]
        target = target_tensor.squeeze(-1).detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted,
            actual=target,
            target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
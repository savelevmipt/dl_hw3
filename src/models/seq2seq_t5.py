import torch
from torch.nn import CrossEntropyLoss, Embedding
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor

import metrics


class Seq2SeqT5(torch.nn.Module):
    def __init__(self, 
                 device,
                 pretrained_name,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 target_tokenizer,
                 start_symbol,
                 lr,
                 are_source_target_tokenizers_same=False):
        super(Seq2SeqT5, self).__init__()
        
        self.target_tokenizer = target_tokenizer

        self.max_len_sent = target_tokenizer.max_len_sent
        self.start_id = self.target_tokenizer.word2index[start_symbol]

        self.device = device
        
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_name).to(self.device)
        self.model.resize_token_embeddings(encoder_vocab_size)
        
        
        if not are_source_target_tokenizers_same: 
            self.model.decoder.set_input_embeddings(Embedding(decoder_vocab_size, self.model.config.d_model).to(self.device))
        
        self.model.set_output_embeddings(torch.nn.Linear(self.model.lm_head.in_features, decoder_vocab_size).to(self.device))
        
        self.crit = CrossEntropyLoss().to(self.device)
        
        self.optim = Adafactor(self.model.parameters(),
                                   lr=lr,
                                   relative_step=False,
                                   warmup_init=False)

    def forward(self, input_tensor: torch.Tensor):
        mem = None
        distr_step, pred_tokens = []

        pred = torch.full((input_tensor.size(0), 1), 
                          self.start_id, 
                          dtype=torch.long, 
                          device=self.device)
        
        for _ in range(self.max_len_sent):
            model_output = self.model(
                input_ids=input_tensor,
                decoder_input_ids=pred,
                encoder_outputs=mem,
                return_dict=True
            )
            mem = (model_output.encoder_last_hidden_state, )
            logits = model_output.logits.transpose(0, 1)[-1]
            __, next_word = torch.max(logits, dim=1)
            pred = torch.cat([pred, next_word.unsqueeze(1)], dim=1)
            pred_tokens.append(next_word.clone().detach().cpu())
            distr_step.append(logits)
        return pred_tokens, distr_step

    def training_step(self, batch):
        input_tensor, target_tensor = batch
        self.optim.zero_grad()

        _pred, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        target_length = target_tensor.shape[1]
        l = 0.0
        for i in range(target_length):
            l += self.crit(
                decoder_outputs[i].squeeze(), target_tensor[:, i, :].squeeze()
            )
        l = l / target_length
        l.backward()
        self.optim.step()
        item = l.item()
        return item

    @torch.no_grad()
    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        _pred, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        with torch.no_grad():
            target_length = target_tensor.shape[1]
            l = 0
            for i in range(target_length):
                l += self.crit(
                    decoder_outputs[i].squeeze(), target_tensor[:, i, :].squeeze()
                )
            l = l / target_length
        item = l.item()
        return item

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    def __init__(self, sentence_list, v_size=4000, pfl_bool=True, max_len_sent=15):
        """
            sentence_list - список предложений для обучения
        """
        self.max_len_sent = max_len_sent
        self.pfl_bool = pfl_bool
        self.tokenizer = Tokenizer(BPE(unk_token="UNK"))
        self.special_tokens = ["PAD", "UNK", "EOS", "SOS"]
        _trainer = BpeTrainer(v_size=v_size,
                             special_tokens=self.special_tokens)
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train_from_iterator(sentence_list, trainer=_trainer)
        self.tokenizer.post_processor = TemplateProcessing(
            single="SOS $A EOS",
            pair="SOS $A EOS $B:1 EOS:1",
            special_tokens=[("SOS", self.tokenizer.token_to_id("SOS")),
                            ("EOS", self.tokenizer.token_to_id("EOS"))],
        )

    def pd_help(self, token_ids_list):
        _len_t = 2 * self.max_len_sent
        if len(token_ids_list) < _len_t: 
            res = token_ids_list + [self.tokenizer.token_to_id("PAD")] * (_len_t - len(token_ids_list))
        else:
            res = token_ids_list[:_len_t - 1] + [self.tokenizer.token_to_id("EOS")]
        return res

    def __call__(self, sentence):
        tokenized_data = self.tokenizer.encode(sentence).ids
        if self.pfl_bool:
            tokenized_data = self.pd_help(tokenized_data)
        return tokenized_data

    def decode(self, token_list):
        """
        token_list - предсказанные ID токенизатора
        """
        tokens = self.tokenizer.decode(token_list).split()
        token_list = []
        for token in tokens:
            if token in self.special_tokens:
                continue
            token_list.append(token)
        return token_list

if __name__ == '__main__':
    pass
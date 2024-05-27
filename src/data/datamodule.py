from torch.utils.data import DataLoader

from data.mt_dataset import MTDataset
from data.space_tokenizer import SpaceTokenizer
from data.bpe_tokenizer import BPETokenizer
from data.utils import TextUtils, short_text_filter_function


class DataManager:
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.input_lang_n_words = None
        self.output_lang_n_words = None
        self.device = device
        self.target_tokenizer = None
        self.source_tokenizer = None

    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"])
        prefix_filter = self.config['prefix_filter']
        if prefix_filter:
            prefix_filter = tuple(prefix_filter)

        source_sentences, target_sentences = [], []
        unique_sources = set()
        for pair in pairs:
            source, target = pair[0], pair[1]
            if short_text_filter_function(pair, self.config['max_length'], prefix_filter) and source not in unique_sources:
                source_sentences.append(source)
                target_sentences.append(target)
                unique_sources.add(source)

        train_size = int(len(source_sentences)*self.config["train_size"])
        source_train_sentences, source_val_sentences = source_sentences[:train_size], source_sentences[train_size:]
        target_train_sentences, target_val_sentences = target_sentences[:train_size], target_sentences[train_size:]

        self.source_tokenizer = BPETokenizer(source_train_sentences, pfl_bool=True, v_size=4000, max_len_sent=self.config['max_length'])
        tokenized_source_train_sentences = [self.source_tokenizer(s) for s in source_train_sentences]
        tokenized_source_val_sentences = [self.source_tokenizer(s) for s in source_val_sentences]

        self.target_tokenizer = BPETokenizer(target_train_sentences, pfl_bool=True, v_size=4000, max_len_sent=self.config['max_length'])
        tokenized_target_train_sentences = [self.target_tokenizer(s) for s in target_train_sentences]
        tokenized_target_val_sentences = [self.target_tokenizer(s) for s in target_val_sentences]

        train_dataset = MTDataset(tokenized_source_list=tokenized_source_train_sentences,
                                  tokenized_target_list=tokenized_target_train_sentences, dev=self.device)

        val_dataset = MTDataset(tokenized_source_list=tokenized_source_val_sentences,
                                tokenized_target_list=tokenized_target_val_sentences, dev=self.device)

        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      batch_size=self.config["batch_size"],
        )

        val_dataloader = DataLoader(val_dataset, shuffle=True,
                                    batch_size=self.config["batch_size"], drop_last=True)
        return train_dataloader, val_dataloader

    def MTDataset(self, tokenized_source_list, tokenized_target_list, dev):
        pass


if __name__ == '__main__':
    import torch
    import yaml
    with open(r'/Users/savelevaleksandr/DL_hw3/configs/data_config.yaml') as myfile:
        cfg = yaml.load(myfile, Loader=yaml.FullLoader)
    dm = DataManager(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = dm.prepare_data()

    for i in train_dataloader:
        print(i)
        break
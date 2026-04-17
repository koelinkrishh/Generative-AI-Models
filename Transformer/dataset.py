import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    """ Pytorch dataset class for dataset arrangement
    """
    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len)->None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = src_tokenizer
        self.tokenizer_tgt = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # Convert special tokens to their corresponding IDs
        self.sos_id = self.tokenizer_tgt.token_to_id("[SOS]")
        self.eos_id = self.tokenizer_tgt.token_to_id("[EOS]")
        self.pad_id = self.tokenizer_tgt.token_to_id("[PAD]")
        
        self.sos_token = torch.tensor([self.sos_id], dtype=torch.long)
        self.eos_token = torch.tensor([self.eos_id], dtype=torch.long)
        self.pad_token = torch.tensor([self.pad_id], dtype=torch.long)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        encode_input_token = self.tokenizer_src.encode(src_text).ids
        encode_target_token = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(encode_input_token) - 2     # 2 extra token consumed for sos and eos tokens
        dec_num_padding_tokens = self.seq_len - len(encode_target_token) - 1    # 1 extra token consumed for sos token
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence length exceeds the maximum sequence length.")
        
        # Add special tokens and padding
        encoder_input = torch.cat([self.sos_token, torch.tensor(encode_input_token, dtype=torch.long), 
            self.eos_token, self.pad_token.repeat(enc_num_padding_tokens)])
        decoder_input = torch.cat([self.sos_token, torch.tensor(encode_target_token, dtype=torch.long), 
            self.pad_token.repeat(dec_num_padding_tokens)])
        
        label = torch.cat([torch.tensor(encode_target_token, dtype=torch.long), 
            self.eos_token, self.pad_token.repeat(dec_num_padding_tokens)])
        
        assert encoder_input.size(0) == self.seq_len, f"Encoder input length {encoder_input.size(0)} does not match the specified sequence length {self.seq_len}."
        assert decoder_input.size(0) == self.seq_len, f"Decoder input length {decoder_input.size(0)} does not match the specified sequence length {self.seq_len}."
        assert label.size(0) == self.seq_len, f"Label length {label.size(0)} does not match the specified sequence length {self.seq_len}."

        return {"encoder_input": encoder_input, "decoder_input": decoder_input, "label": label,
            "encoder_mask":(encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0), # (Batch=1, seq_dim=1, seq_len)
            # (1, seq_len) & (1, seq_len, seq_len)
            "decoder_mask":(decoder_input != self.pad_id).unsqueeze(0).unsqueeze(0) & self.causal_mask(decoder_input.size(0)),
            "source_text":src_text, "target_text":tgt_text,
        }
    """
    Encoder mask: mark padding token which are added to level all sentences in batch to equal length
    Decoder mask: 1) Autoregressive mask -> prevent model from looking at future tokens
                  2) Padding mask -> prevent model from looking at padding tokens
    """
        
    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        return mask==0



if __name__ == "__main__":
    print(TranslationDataset.causal_mask(6))
    
    
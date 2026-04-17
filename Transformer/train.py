import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import TranslationDataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path
from config import get_config, get_weights_file_path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Turning off warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TensorFlow logging (if using TensorFlow backend for any reason)
os.environ["IF_ENABLE_ONEDNN_OPTS"] = "0" # Disable oneDNN optimizations to suppress related warnings




def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
   sos_idx = tokenizer_tgt.token_to_id('[SOS]')
   eos_idx = tokenizer_tgt.token_to_id('[EOS]')
   
   # Precompute the encoder output and reuse it for every token of decoder
   encoder_output = model.encode(source, source_mask)
   ## Decoder working -> Takes all token upto current token and gives next token
   
   # Initialize the decoder input with SOS token
   decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # (Batch=1, seq_len=1)
   
   while True:
      if decoder_input.size(1) == max_len:
         break
      
      # Build mask for the target
      decoder_mask = TranslationDataset.causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
      # Calculate decoder output
      out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
      
      # Get the prob for next token
      prob = model.project(out[:, -1]) # Greedy approach uses only the last token's output to predict the next token. (Batch=1, target_vocab_size)
      _, next_word = torch.max(prob, dim=-1) # [max prob]
      # Append the predicted token to the decoder input for the next iteration
      decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1) # (Batch=1, seq_len+1)
      
      if next_word == eos_idx:
         break
   
   return decoder_input
   


def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device,
   print_msg, global_state, writer, num_examples=2):
   model.eval() # Put model in evaluation mode
   
   count = 0
   source_text = []
   expected = []
   predicted = []
   
   # Size of the control window
   console_width = 80
   
   with torch.no_grad():
      for batch in validation_dataset:
         count += 1
         encoder_input = batch['encoder_input'].to(device)
         encoder_mask = batch['encoder_mask'].to(device)
         
         assert encoder_input.size(0) == 1, "Batch size should be 1 for validation."
         
         model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
         
         model_out_list = model_out[0].cpu().tolist()
         predicted_text = tokenizer_tgt.decode(model_out_list)

         curr = (batch['source_text'][0], batch['target_text'][0], predicted_text)
         source_text.append(curr[0])
         expected.append(curr[1])
         predicted.append(curr[2])
         
         # Print progress
         print_msg("-"*console_width)
         print_msg(f"EXAMPLE {count}/{num_examples}")
         print_msg(f"SOURCE: {curr[0]}")
         print_msg(f"EXPECTED: {curr[1]}")
         print_msg(f"PREDICTED: {curr[2] if curr[2] != '' else '[EMPTY STRING / ONLY SPECIAL TOKENS]'}")
         print_msg(f"RAW TOKENS: {model_out_list}")
         
         if count == num_examples:
            break
      
   # Log the validation examples in Tensorboard
   if writer is not None:
      for i in range(num_examples):
         writer.add_text(f"Validation example {i+1}", f"Source: {source_text[i]}\nExpected: {expected[i]}\nPredicted: {predicted[i]}", global_state)
      writer.flush()
         



### Tokenizer: Splitting the text into tokens
def get_all_sentences(ds, lang):
   """Generator that yields sentences from the dataset for a given language."""
   for item in ds:
      yield item['translation'][lang]
   # return ds[lang]['train']['text'] + ds[lang]['validation']['text'] + ds[lang]['test']['text']

def get_or_build_tokenizer(config, ds, lang):
   """ Download or load pre-built tokenizer for the given language. """
   tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))
   
   if not tokenizer_path.exists():
      print(f"Building tokenizer for {lang}...")
      tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
      tokenizer.pre_tokenizer = Whitespace()
      trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"], min_frequency=2)
      tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
      tokenizer.save(str(tokenizer_path))
   else:
      tokenizer = Tokenizer.from_file(str(tokenizer_path))
      print(f"Loaded tokenizer for {lang} from {tokenizer_path}")
      
   return tokenizer

def get_dataset(config):
   """ Load the dataset for the given language pair. """
   # ds = load_dataset(config['dataset_name'], config['dataset_config'])
   ds_raw = load_dataset(config['dataset_name'], split="train")
   ## Using smaller fraction for testing
   ds_raw = ds_raw.select(range(2000))
   
   # Build tokenizers
   tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
   tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])
   
   # Computing maximum sent length
   # max_src_sent_length = 0
   # max_tgt_sent_length = 0
   
   # for item in ds_raw:
   #     src_tokens = tokenizer_src.encode(item['translation'][config['src_lang']]).ids
   #     tgt_tokens = tokenizer_tgt.encode(item['translation'][config['tgt_lang']]).ids
   #     max_src_sent_length = max(max_src_sent_length, len(src_tokens))
   #     max_tgt_sent_length = max(max_tgt_sent_length, len(tgt_tokens))
   
   # # Add special token space
   # max_src_sent_length += 2
   # max_tgt_sent_length += 2
   
   # Splitting dataset
   train_ds_size = int(0.9 * len(ds_raw))
   val_ds_size = len(ds_raw) - train_ds_size
   train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])
   
   Train_dataset = TranslationDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
   Validation_dataset = TranslationDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
   
   train_dataloader = DataLoader(Train_dataset, batch_size=config['batch_size'], shuffle=True)
   val_dataloader = DataLoader(Validation_dataset, batch_size=1, shuffle=True)
   
   return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt#, max_src_sent_length, max_tgt_sent_length

import Transformer

def get_model(config, source_vocab_size, target_vocab_size):
   """ Initialize the Transformer model. """
   # model = TF.Transformer(
   #     num_encoder_layers=config['num_encoder_layers'],
   #     num_decoder_layers=config['num_decoder_layers'],
   #     d_model=config['d_model'],
   #     num_heads=config['num_heads'],
   #     d_ff=config['d_ff'],
   #     source_vocab_size=source_vocab_size,
   #     target_vocab_size=target_vocab_size,
   #     max_seq_length=config['max_seq_length']
   # )
   
   model = Transformer.build_model(source_vocab_size, target_vocab_size,
      config['seq_len'], config['seq_len'], config['d_model'], config['d_ff'],
      config['num_heads'], config['num_encoder_layers'], config['num_decoder_layers'],
      config['dropout'])
   
   return model

## Main training loop
def train_model(config):   
   # Defining device
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   
   Path(config['model_folder']).mkdir(parents=True, exist_ok=True,)
   
   # Get dataset and tokenizers
   train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
   # Get model
   model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
   
   # Tensorboard -> Visualize training process
   writer = SummaryWriter(config['experiment_name'])
   
   optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
   
   initial_epoch = 0
   global_step = 0
   
   # Reload trained state
   if config['preload'] is not None:
      model_filename = get_weights_file_path(config, config["preload"])
      print(f"Loading preloaded model from {model_filename}...")
      
      state = torch.load(model_filename)
      initial_epoch = state['epoch'] + 1 # same epoch + 1
      global_step = state['global_step'] # same global step
      
      optimizer.load_state_dict(state['optimizer']) # same adam history
      
      # model.load_state_dict(torch.load(config['preload']))
      
   ## Loss function should not be accounted for padding tokens
   loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device) # From target tokenzier as loss is calculated for target tokens mismatch
   # label_smoothing -> Add uncertainty to the model's predictions, which can help prevent overfitting and improve generalization.
   
   for ep in range(initial_epoch, config['num_epochs']): # remaining epochs
      model.train() # set model to training
      batch_iter = tqdm(train_dataloader, desc=f"Processing epoch: {ep+1:02d}/{config['num_epochs']:02d}", unit="batch")
      
      total_loss = 0
      for batch in batch_iter: # (Batch, seq_len)
         # STEP 1: Move batch to device
         encoder_input = batch['encoder_input'].to(device) # (Batch, seq_len)
         decoder_input = batch['decoder_input'].to(device) # (Batch, seq_len)
         label = batch['label'].to(device) # (Batch, seq_len)
         ## Padding hidder mask
         encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, seq_len)
         ## Causal mask for autoregressive nature
         decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, seq_len)
         
         # STEP 2: Forward pass
         encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, seq_len, d_model)
         decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, seq_len, d_model)
         projection_output = model.project(decoder_output) # (Batch, seq_len, target_vocab_size)
         
         # STEP 3: Loss calculation
         # Need to transform shape of output: (B, seq_len, tgt_vocab_size) -> (B*seq_len, tgt_vocab_size)
         loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
         batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})
         
         # Log the loss at current iteration
         writer.add_scalar("train loss", loss.item(), global_step)
         writer.flush()
         
         # STEP 4: Backpropagation
         loss.backward()
         
         # STEP 5: Update the weights
         optimizer.step()
         optimizer.zero_grad()
         
         global_step += 1
      
      # Run validation at the end of each epoch
      run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
         lambda msg: batch_iter.write(msg), global_step, writer, num_examples=2)
         
      # Save the trained model at the end of each epoch
      model_filename = get_weights_file_path(config, f"{ep:02d}")
      # Also save model state configuration statistics like optimizer state
      torch.save({'epoch':ep, 'model_state_dict':model.state_dict(),
         'optimizer':optimizer.state_dict(), 'global_step':global_step
      }, model_filename)
      

if __name__ == "__main__":
   import warnings
   
   warnings.filterwarnings("ignore") # Ignore all warnings
   config = get_config()
   train_model(config)
         


from typing import Optional, List, Tuple
import torch
from torch import nn
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import Modelargs, Transformer




class LLama(nn.Module):
    def __init__(self, model:Transformer, tokenizer:SentencePieceProcessor, args:Modelargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        
    @staticmethod
    def _compute_hidden_dim(args:Modelargs):
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        else:
            hidden_dim = int((2/3) * 4 * args.dim)
        return ((hidden_dim + args.multiple_of - 1) // args.multiple_of) * args.multiple_of

    @staticmethod
    def _slice_state_dict(state_dict:dict, args:Modelargs):
        new_sd = {}
        new_dim = args.dim
        new_hidden = LLama._compute_hidden_dim(args)
        orig_dim = state_dict['tok_embeddings.weight'].shape[1]
        assert new_dim <= orig_dim, 'Target dimension must be smaller than the checkpoint dimension'

        new_sd['tok_embeddings.weight'] = state_dict['tok_embeddings.weight'][:, :new_dim].clone()
        new_sd['output.weight'] = state_dict['output.weight'][:, :new_dim].clone()
        new_sd['norm.weight'] = state_dict['norm.weight'][:new_dim].clone()

        for layer_idx in range(args.n_layers):
            prefix = f'layers.{layer_idx}.'
            new_sd[prefix + 'attention.wq.weight'] = state_dict[prefix + 'attention.wq.weight'][:new_dim, :new_dim].clone()
            new_sd[prefix + 'attention.wk.weight'] = state_dict[prefix + 'attention.wk.weight'][:new_dim, :new_dim].clone()
            new_sd[prefix + 'attention.wv.weight'] = state_dict[prefix + 'attention.wv.weight'][:new_dim, :new_dim].clone()
            new_sd[prefix + 'attention.wo.weight'] = state_dict[prefix + 'attention.wo.weight'][:new_dim, :new_dim].clone()
            new_sd[prefix + 'feed_forward.w1.weight'] = state_dict[prefix + 'feed_forward.w1.weight'][:new_hidden, :new_dim].clone()
            new_sd[prefix + 'feed_forward.w2.weight'] = state_dict[prefix + 'feed_forward.w2.weight'][:new_dim, :new_hidden].clone()
            new_sd[prefix + 'feed_forward.w3.weight'] = state_dict[prefix + 'feed_forward.w3.weight'][:new_hidden, :new_dim].clone()
            new_sd[prefix + 'attention_norm.weight'] = state_dict[prefix + 'attention_norm.weight'][:new_dim].clone()
            new_sd[prefix + 'ffn_norm.weight'] = state_dict[prefix + 'ffn_norm.weight'][:new_dim].clone()

        return new_sd

    @staticmethod
    def build(checkpoint_dir:str, tokenizer_path:str, load_model:bool, max_seq_len:int, max_batch_size:int, device:str, test_mode:bool=False, partial_layers:Optional[int]=None, partial_heads:Optional[int]=None):
        base_dir = Path(__file__).resolve().parent
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.is_absolute():
            checkpoint_path = base_dir / checkpoint_path
        params_path = checkpoint_path / "params.json"
        assert params_path.exists(), f"params.json not found in {checkpoint_path}"
        
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.is_absolute():
            tokenizer_path = base_dir / tokenizer_path
        if not tokenizer_path.exists():
            alt_tokenizer = checkpoint_path / "tokenizer.model"
            if alt_tokenizer.exists():
                tokenizer_path = alt_tokenizer
        assert tokenizer_path.exists(), f"tokenizer.model not found at {tokenizer_path} or inside {checkpoint_path}"
        
        print(f"Resolved checkpoint path: {checkpoint_path}")
        print(f"Resolved tokenizer path: {tokenizer_path}")
        
        checkpoint = None
        prev_time = time.time()
        if load_model and not test_mode:
            checkpoint_files = sorted(checkpoint_path.glob('*.pth'), key=lambda x: x.stat().st_mtime, reverse=True)
            assert len(checkpoint_files) > 0, f"No checkpoint found in {checkpoint_path}"
            chk_path = checkpoint_files[0]
            print(f"Loading checkpoint: {chk_path}")
            checkpoint = torch.load(chk_path, map_location=device)
            
            print(f"Found checkpoint in {(time.time() - prev_time):.3f} seconds")
            prev_time = time.time()
            
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
            
        model_args = Modelargs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        
        tokenizer = SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))
        model_args.vocab_size = tokenizer.vocab_size()
        
        if partial_heads is not None:
            base_heads = params.get('n_heads', model_args.n_heads)
            head_dim = params['dim'] // base_heads
            model_args.n_heads = partial_heads
            model_args.dim = head_dim * partial_heads
            model_args.n_kv_head = None
            print(f"Using partial model heads={partial_heads}, dim={model_args.dim}")

        if partial_layers is not None:
            model_args.n_layers = partial_layers
            print(f"Using partial model layers={partial_layers}")
        
        if test_mode:
            print("Building a small test model for inference validation.")
            model_args.dim = min(model_args.dim, 128)
            model_args.n_layers = min(model_args.n_layers, 2)
            model_args.n_heads = min(model_args.n_heads, 4)
            model_args.max_seq_len = min(model_args.max_seq_len, 64)
            model_args.max_batch_size = min(model_args.max_batch_size, 4)
            model_args.n_kv_head = None
        
        # Use explicit dtypes for tensors and move the model to the target device.
        model = Transformer(model_args).to(device)
        
        if load_model:
            if "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if "rope.freqs" in checkpoint:
                del checkpoint["rope.freqs"]
            if partial_layers is not None or partial_heads is not None:
                checkpoint = LLama._slice_state_dict(checkpoint, model_args)
                strict_load = False
            else:
                strict_load = True
            model.load_state_dict(checkpoint, strict=strict_load)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f}s")
        
        return LLama(model, tokenizer, model_args)
    
    def text_completion(self, prompts:List[str], device:Optional[str]=None, temperature:float=0.6, top_p:float=0.9, max_gen_len:Optional[int]=100):
        if device is None:
            device = self.args.device
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len-1
            
        # Convert each prompt into tokens using tokenizer
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        bz = len(prompt_tokens)
        assert bz <= self.args.max_batch_size, f"Batch size {bz} exceeds model's max batch size {self.args.max_batch_size}"
        
        max_prompt_length = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_length <= self.args.max_seq_len, f"Prompt length {max_prompt_length} exceeds model's max sequence length {self.args.max_seq_len}"
        
        total_length = min(max_prompt_length + max_gen_len, self.args.max_seq_len)
        
        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id= self.tokenizer.pad_id()
        tokens = torch.full((bz, total_length), pad_id, dtype=torch.long, device=device)
        
        for k,t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            
        eos_reacherd = torch.Tensor([False]*bz, device=device)
        prompt_tokens_mask = (tokens != pad_id) # True if token is part of the prompt, False if it's padding
        
        for curr in tqdm(range(1,total_length), desc="Generation Tokens"):
            with torch.no_grad():
                logits = self.model(tokens[:, curr-1:curr], start_pos=curr-1)
                logits = logits[:, -1] # only select last token's logit for next generation step
                
            if temperature != 1.0 and temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_from_top_p(probs, top_p=top_p)
            else: # Greedy search
                next_token = torch.argmax(logits[:, -1], dim=-1)
                
            next_token = next_token.reshape(-1)
            """ Padding Tokens: The model initializes a tensor filled with padding tokens for the total length of the sequence.
                Masking: A mask is used to distinguish between existing prompt tokens and new tokens that the model needs to generate.
                    The code checks if a position is a padding token;
                    if it is, that is where the model inserts a newly generated token.
                    If it is a prompt token, the model does not change the token.
            """
            next_token = torch.where(prompt_tokens_mask[:, curr], tokens[:, curr], next_token)
            tokens[:, curr] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached = (~prompt_tokens_mask[:, curr]) & (next_token == self.tokenizer.eos_id()) # Originally padding token + now EOS
            if eos_reached.all(): # Stop generation when all batch have reached EOS
                break
            
        out_tokens = []
        out_text = []
        """
        for prompt_index, curr_prompt_tokens in enumerate(prompt_tokens):
            gen_tokens = tokens[prompt_index, len(curr_prompt_tokens):len(curr_prompt_tokens)+max_gen_len].tolist()
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            out_tokens.append(gen_tokens)
            out_text.append(gen_text)
        ## OR
        """
        for curr_prompt_tokens in tokens.tolist():
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in curr_prompt_tokens:
                eos_index = curr_prompt_tokens.index(self.tokenizer.eos_id())
                curr_prompt_tokens = curr_prompt_tokens[:eos_index]
            
            # Waste to decode special tokens
            filtered_tokens = [t for t in curr_prompt_tokens if t not in {self.tokenizer.bos_id(), self.tokenizer.eos_id(), self.tokenizer.pad_id()}]
            gen_text = self.tokenizer.decode(filtered_tokens)
            out_tokens.append(curr_prompt_tokens)
            out_text.append(gen_text)
        
        return out_tokens, out_text
    
    def sample_from_top_p(self, probs:torch.Tensor, top_p:float):
        probs_sorted, idx = torch.sort(probs, descending=True) # sorted probabilities and their corresponding token indices
        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
        
        mask = cumulative_probs > top_p
        mask[..., 0] = False
        mask = mask.cumsum(dim=-1) > 0
        probs_sorted[mask] = 0.0 # zero out tokens that are outside the top-p threshold
        
        # Renormalize the probabilities after masking
        probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
        
        next_token = torch.multinomial(probs_sorted, num_samples=1)
        # Map back to original token indices
        next_token = torch.gather(idx, dim=-1, index=next_token)
        
        return next_token # Index for next token to generate
        

    
if __name__ == "__main__":
    torch.manual_seed(0)
    
    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
    
    prompts = ["What is the capital of France?"]
    
    model = LLama.build(
        checkpoint_dir="llama-2-7b", 
        tokenizer_path="tokenizer.model",
        load_model=False, test_mode=True,
        max_seq_len=64, max_batch_size=len(prompts), 
        device=device,
    )
    
    # Inference the model with the prompts
    out_tokens, out_text = model.text_completion(prompts, device=device, temperature=0.6, top_p=0.9, max_gen_len=64)
    assert len(out_tokens) == len(prompts)
    
    for i in range(len(prompts)):
        print(f"{out_text[i]}")
        print('-'*50)
    
    print("All Working")
        
        
        
        
        

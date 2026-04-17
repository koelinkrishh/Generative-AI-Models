


# %%
# 3a. Encoder Self-Attention Visualization
get_all_attention_maps("encoder", layers, heads, encoder_input_tokens, encoder_input_tokens, min(20, sentence_len))

# %%
# 3b. Decoder Self-Attention Visualization
get_all_attention_maps("decoder", layers, heads, decoder_input_tokens, decoder_input_tokens, min(20, sentence_len))

# %%
# 3c. Encoder-Decoder Cross-Attention Visualization
get_all_attention_maps("encoder-decoder", layers, heads, encoder_input_tokens, decoder_input_tokens, min(20, sentence_len))

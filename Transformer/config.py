## Main configuration file to save configurations
from pathlib import Path

file_path = Path(__file__).resolve().parent

## Lower config for testing
def get_config():
    return {
        "dataset_name": "cfilt/iitb-english-hindi",
        "batch_size": 32, "num_epochs": 10,
        "lr": 1e-3,
        "seq_len": 100, # specific length for each dataset
        "d_model": 256, "d_ff": 512,
        "src_lang": "en", "tgt_lang": "hi",
        "model_folder": f"{file_path}/weights", "model_basename": "transformer_model",
        "preload":None, # "weights/transformer_model.pt" or None
        "tokenizer_file": f"{file_path}/tokenizer_{{lang}}.json",
        "experiment_name": f"{file_path}/runs/en-hi_transformer",
        "num_heads": 4,
        "num_encoder_layers": 2, "num_decoder_layers": 3,
        "dropout": 0.2
    }

# def get_config():
#     return {
#         "dataset_name": "cfilt/iitb-english-hindi",
#         "batch_size": 16, "num_epochs": 10,
#         "lr": 1e-3,
#         "seq_len": 350, # specific length for each dataset
#         "d_model": 512, "d_ff": 2048,
#         "src_lang": "en", "tgt_lang": "hi",
#         "model_folder": "weights", "model_basename": "transformer_model.pt",
#         "preload":None, # "weights/transformer_model.pt" or None
#         "tokenizer_file": "tokenizer.json",
#         "experiment_name": "runs/en-hi_transformer",
#         "num_heads": 8,
#         "num_encoder_layers": 3, "num_decoder_layers": 5,
#         "dropout": 0.1
#     }


def get_weights_file_path(config, epoch:str):
    folder = config['model_folder']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    
    return str(Path(".")/folder/model_filename)
    
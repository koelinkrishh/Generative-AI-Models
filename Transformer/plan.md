# Coding Transformer

![alt text](https://media.licdn.com/dms/image/v2/D5612AQHl7OAsf21dnQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1695664075976?e=2147483647&v=beta&t=cSxyV0i3c83zl7agvGhvj7TXcESK1RCMXKNwg5LCqJU)


#### 1. Basic Preprocessing:
- Chunking paragraph
- Tokenization
- Padding and Truncation
- Creating Batches

#### 2. Basic functionality of Transformer:
- Embedding
- Positional Encoding
- linear flattening layer
- Softmax output

#### 3. Basic operational layers:
- Attention layer
- Layer normalization
- Feed forward network
- Residual connection

#### 4. Block architecture:
   ##### Encoder Block
      - Multi-head Self-attention
      - Feed forward network
      - Layer normalization
      - Residual connection
   ##### Decoder Block
      - Masked Multi-head Self-attention
      - Multi-head Cross-attention
      - Feed forward network
      - Layer normalization
      - Residual connection

# Setting up for Translation task:
We will train our transformer on English to hindi translation task. We will use the IWSLT 2016 dataset for this purpose.
It contains around 200k sentence pairs for training and around 7k sentence pairs for validation and testing.

#### 5. Getting dataset
- Loading dataset from hugging face datasets library
- Setup dataclass from dataset to load sentences in batches
- Setup tokenizer to convert sentences to tokens
- Setup dataloader to load data in batches



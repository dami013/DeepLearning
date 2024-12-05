"""
Assignment 3 - Language Models with LSTM
Author: Damiano Ficara
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torch import nn
from torch import optim
from torch.functional import F



'''
Utility Function
'''
def keys_to_values(keys, map, default_if_missing=None):
   """Convert keys to their corresponding values in the mapping.
   Args:
       keys: Input sequence to convert 
       map: Dictionary with mapping relationships
       default_if_missing: Value to use for missing keys
   """
   return [map.get(key, default_if_missing) for key in keys]

'''
Q6 - Dataset class for managing news headline sequences
'''
class NewsSDataset(Dataset):
    def __init__(self, tokenized_sequences, word_to_int):
        """
        Args:
            tokenized_sequences: List of word sequences
            word_to_int: Dictionary mapping words to integers
        """
        self.sequences = []

        # Transform word sequences into integer sequences using vocabulary mapping
        for seq in tokenized_sequences:
            int_seq = [word_to_int.get(word, word_to_int['<EOS>']) for word in seq]
            self.sequences.append(int_seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Returns (x, y) where:
           x = input sequence (all tokens except last)
           y = target sequence (all tokens except first)
        """
        sequence = self.sequences[idx]
        return (
            torch.tensor(sequence[:-1]),  # Input sequence for model
            torch.tensor(sequence[1:])    # Target sequence for prediction
        )
def collate_fn(batch, pad_value):
    """
    Pads sequences in batch to same length for efficient processing
    Args:
        batch: List of (x, y) tuples from dataset
        pad_value: Integer index for PAD token
    """
    # Split input and target sequences
    x_seqs, y_seqs = zip(*batch)

    # Pad all sequences to match the longest sequence in the batch
    x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=pad_value)
    y_padded = pad_sequence(y_seqs, batch_first=True, padding_value=pad_value)

    return x_padded, y_padded
'''
Q7 - LSTM Model Implementation
'''
class TextModel(nn.Module):
   """Basic LSTM architecture for language modeling tasks.
   Takes integer token sequences as input and predicts next tokens.
   """
   def __init__(self, map, hidden_size=1024, emb_dim=150, n_layers=1):
       """Initialize model architecture with embedding, LSTM and output layers"""
       super(TextModel, self).__init__()
       # Save model configuration
       self.vocab_size = len(map)  # Total number of unique tokens
       self.hidden_size = hidden_size  # LSTM hidden state dimension
       self.emb_dim = emb_dim  # Word embedding dimension
       self.n_layers = n_layers  # Number of LSTM layers

       # Embedding layer converts token IDs to dense vectors
       self.embedding = nn.Embedding(
           num_embeddings=self.vocab_size,
           embedding_dim=self.emb_dim,
           padding_idx=map["PAD"])  # Handle padding tokens

       # LSTM processes embedded sequences 
       self.lstm = nn.LSTM(
           input_size=self.emb_dim,
           hidden_size=self.hidden_size,
           num_layers=self.n_layers,
           batch_first=True)  # Expect batch dimension first

       # Linear layer maps LSTM outputs to vocabulary scores
       self.fc = nn.Linear(
           in_features=self.hidden_size,
           out_features=self.vocab_size)

   def forward(self, x, prev_state):
       """Forward pass: embed tokens, process with LSTM, project to vocab"""
       embed = self.embedding(x)  # Convert tokens to vectors
       yhat, state = self.lstm(embed, prev_state)  # Process sequence
       out = self.fc(yhat)  # Get vocabulary predictions 
       return out, state

   def init_state(self, b_size=1):
       """Initialize LSTM state tensors to zeros"""
       return (torch.zeros(self.n_layers, b_size, self.hidden_size),  # Hidden state
               torch.zeros(self.n_layers, b_size, self.hidden_size))  # Cell state
'''
Text Generation Functions
'''

def random_sample_next(model, x, prev_state, topk=5, uniform=True):
   """Generate next token using random sampling from top-k predictions."""
   # Generate predictions and select next token using random sampling
   out, state = model(x.to(DEVICE), prev_state=tuple(s.to(DEVICE) for s in prev_state))
   last_out = out[0, -1, :]  # Get predictions for last position
   
   # Select top-k most likely tokens
   topk = topk if topk else last_out.shape[0]
   top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)
   
   # Calculate sampling probabilities 
   p = None if uniform else F.softmax(top_logit, dim=-1).cpu().detach().numpy()
   
   # Sample next token from top-k predictions
   sampled_ix = np.random.choice(top_ix.cpu().numpy(), p=p)
   
   return sampled_ix, state

def sample_argmax(model, x, prev_state):
   """Generate next token using greedy selection (highest probability)."""
   # Generate predictions and select next token using greedy strategy
   out, state = model(x.to(DEVICE), prev_state=tuple(s.to(DEVICE) for s in prev_state))
   last_pred = out[:, -1, :]  # Get predictions for last position
   next_word = torch.argmax(last_pred, dim=-1).item()  # Select most likely token
   
   return next_word, state

def sample(model, prompt, stop_on, sample_method='random', topk=5, max_seqlen=18):
   """Generate text sequence from prompt using specified sampling method.
   Continues until max length or EOS is reached."""
   prompt = prompt if isinstance(prompt, (list, tuple)) else [prompt]

   model.eval()
   with torch.no_grad():
       sampled_ix_list = prompt[:]
       x = torch.tensor([prompt])
       prev_state = model.init_state(b_size=1)

       # Generate tokens until max length or stop token
       for t in range(max_seqlen - len(prompt)):
           if sample_method == 'argmax':
               # Use greedy selection
               sampled_ix, prev_state = sample_argmax(model, x, prev_state)
           else:
               # Use random sampling from top-k
               sampled_ix, prev_state = random_sample_next(model, x, prev_state, topk=topk)

           sampled_ix_list.append(sampled_ix)
           x = torch.tensor([[sampled_ix]])

           if sampled_ix == stop_on:
               break

   model.train()
   return sampled_ix_list
'''
Standard Training Implementation
'''

def train(model, data, num_epochs, criterion, lr=0.001, print_every=50, clip=None):
   """Train the language model with gradient clipping and progress monitoring."""
   # Setup model and training environment
   DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(DEVICE)
   criterion = criterion.to(DEVICE)
   model.train()

   # Initialize tracking metrics and optimizer
   loss_hist = []  # Track loss over epochs
   generated_text_list = []  # Store sample generations
   perplexity_hist = []  # Track model perplexity
   optimizer = optim.Adam(model.parameters(), lr=lr)
   
   total_batches = len(data)
   epoch = 0

   # Main training loop over epochs
   while epoch < num_epochs:
       epoch += 1
       epoch_loss = 0.0

       # Process data batches
       for batch_idx, (x, y) in enumerate(data, 1):
           x = x.to(DEVICE)
           y = y.to(DEVICE)
           optimizer.zero_grad()

           # Setup LSTM state for batch
           prev_state = model.init_state(b_size=x.shape[0])
           prev_state = tuple(s.to(DEVICE) for s in prev_state)

           # Forward pass through model
           out, state = model(x, prev_state=prev_state)
           loss_out = out.permute(0, 2, 1)  # Reshape for loss calculation
           loss = criterion(loss_out, y)
           epoch_loss += loss.item()

           # Optimization step with optional gradient clipping
           loss.backward()
           if clip:
               nn.utils.clip_grad_norm_(model.parameters(), clip)
           optimizer.step()

       # Update tracking metrics
       avg_epoch_loss = epoch_loss / total_batches
       loss_hist.append(avg_epoch_loss)
       perplexity = torch.exp(torch.tensor(avg_epoch_loss))
       perplexity_hist.append(perplexity.item())

       # Progress reporting and text generation
       if print_every and (epoch % print_every) == 0:
           print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:8.4f}, Perplexity: {perplexity:8.4f}")
           generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'], sample_method='argmax')
           generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
           generated_text_list.append(generated_text)
           print(f"Generated text: {generated_text}\n")

       # Early stopping check when it reaches the threshold of 1.5
       if avg_epoch_loss < 1.5:
           print(f"\nTarget loss of 1.5 reached at epoch {epoch}!")
           break

   # Display generation progress
   if len(generated_text_list) > 0:
       print("Beginning list:", generated_text_list[0])
       middle_index = len(generated_text_list) // 2
       print("Middle list:", generated_text_list[middle_index])
       print("End list:", generated_text_list[-1])

   return model, loss_hist, perplexity_hist
'''
Truncated Backpropagation Through Time (TBPTT) Training Implementation
'''
def train_tbbtt(model, data, num_epochs, criterion, chunk_len, lr=0.001, print_every=50, clip=None):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    model.train()
    
    # Initialize tracking metrics
    loss_hist = []
    generated_text_list = []
    perplexity_hist = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_batches = len(data)
    epoch = 0
    
    # Training loop with TBPTT
    while epoch < num_epochs:
        epoch += 1
        epoch_loss = 0.0
        
        # Process each batch
        for batch_idx, (x, y) in enumerate(data, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Get sequence length for current batch
            seq_length = x.size(1)
            
            # Initialize LSTM state
            current_state = model.init_state(b_size=x.shape[0])
            current_state = tuple(s.to(DEVICE) for s in current_state)
            
            # Process sequence in chunks for TBPTT
            chunk_losses = []
            for chunk_start in range(0, seq_length, chunk_len):
                optimizer.zero_grad()
                
                # Get chunk boundaries
                chunk_end = min(chunk_start + chunk_len, seq_length)
                
                # Extract current chunk
                chunk_x = x[:, chunk_start:chunk_end]
                chunk_y = y[:, chunk_start:chunk_end]
                
                # Forward pass on chunk
                chunk_out, new_state = model(chunk_x, prev_state=current_state)
                loss_out = chunk_out.permute(0, 2, 1)
                
                # Compute loss for current chunk
                loss = criterion(loss_out, chunk_y)
                chunk_losses.append(loss.item())
                
                # Backward pass and optimization
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
                # Update state for next chunk
                current_state = tuple(s.detach() for s in new_state)
            
            # Average chunk losses for this batch
            batch_loss = sum(chunk_losses) / len(chunk_losses)
            epoch_loss += batch_loss
        
        # Calculate and store metrics
        avg_epoch_loss = epoch_loss / total_batches
        loss_hist.append(avg_epoch_loss)
        perplexity = torch.exp(torch.tensor(avg_epoch_loss))
        perplexity_hist.append(perplexity.item())
        
        # Print progress and generate sample text
        if print_every and (epoch % print_every) == 0:
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:8.4f}, Perplexity: {perplexity:8.4f}")
            generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'], sample_method='argmax')
            generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
            generated_text_list.append(generated_text)
            print(f"Generated text: {generated_text}\n")
        
        # Early stopping check
        if avg_epoch_loss < 1.5:
            print(f"\nTarget loss of 1.5 reached at epoch {epoch}!")
            break
    
    # Print generation progress examples
    if len(generated_text_list) > 0:
        print("\nGeneration progress throughout training:")
        print("Beginning:", generated_text_list[0])
        middle_index = len(generated_text_list) // 2
        print("Middle:", generated_text_list[middle_index])
        print("End:", generated_text_list[-1])
    
    return model, loss_hist, perplexity_hist

'''
Visualization Function
'''
def plot_metrics(loss_hist, perplexity_hist, title_prefix="", save_path=None):
    """Plot training metrics to visualize model progress."""
    plt.figure(figsize=(12, 5))

    # Plot loss history
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist, label='Training Loss')
    plt.axhline(y=1.5, color='r', linestyle='--', label='Target Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix}Training Loss Over Time')
    plt.legend()
    plt.grid(True)

    # Plot perplexity history
    plt.subplot(1, 2, 2)
    plt.plot(perplexity_hist, label='Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(f'{title_prefix}Perplexity Over Time')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()




if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    # Probably, this below must be changed if you work with a M1/M2/M3 Mac
    torch.cuda.manual_seed(seed) # for CUDA
    torch.backends.cudnn.deterministic = True # for CUDNN
    torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    # Set device based on available hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps'
        if torch.backends.mps.is_available() else 'cpu')

    '''
    Q1 - Load the news category dataset
    '''
    ds = load_dataset("heegyu/news-category-dataset")
    print(ds['train'])
    print(ds['train'].column_names)
    '''
    Q2 - Filter dataset to only include POLITICS category
    '''
    ds = ds['train'].filter(lambda x: x['category'] == 'POLITICS')
    print(len(ds))

    '''
    Q3 - Tokenize headlines and add EOS token
    '''
    tokenized_text = []
    for title in ds['headline']:
        words = title.lower().split()
        words.append('<EOS>')  # Add end-of-sequence token
        tokenized_text.append(words)
    '''
    Q4 - Build vocabulary and find most common words
    '''
    # Count word frequencies
    word_freq = {}
    for sequence in tokenized_text:
        for word in sequence:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort words by frequency for vocabulary creation
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Create mappings between words and integers
    word_to_int = {'<EOS>': 0}  # Start with special tokens
    current_idx = 1

    for word, freq in sorted_words:
        if word != '<EOS>':  # Skip EOS token as already added
            word_to_int[word] = current_idx
            current_idx += 1

    word_to_int['PAD'] = len(word_to_int)  # Add padding token last

    # Create reverse mapping (integer to word)
    int_to_word = {i: word for word, i in word_to_int.items()}

    # Find and print most common words
    top_5_words = [(word, freq) for word, freq in sorted_words[:6]
                   if word not in ['<EOS>', 'PAD']]

    print("5 most common words:")
    for word, freq in top_5_words:
        print(f"{word}: {freq} occurrences")

    print(f"\nTotal vocabulary size (including special tokens): {len(word_to_int)}")

    '''
    Q5-Q6 - Create dataset and dataloader instances
    '''
    # Initialize dataset with batch processing
    batch_size = 32
    dataset = NewsSDataset(tokenized_text, word_to_int)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                          collate_fn=lambda b: collate_fn(b, word_to_int["PAD"]),
                          shuffle=True)

    '''
    Initialize base model for evaluation
    '''
    # Create model instance
    model = TextModel(word_to_int).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])

    '''
    Evaluate untrained model performance
    '''
    # Test model with sample prompt
    prompt_text = "the president wants"
    prompt_indices = keys_to_values(prompt_text.split(), word_to_int)
    print("\nInitial prompt:", prompt_text)

    # Test random sampling generation
    print("\nGenerated sequences using random sampling (topk=5):")
    for i in range(3):
        generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'],
                                 sample_method='random', topk=5)
        generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
        print(f"Sequence {i+1}: {generated_text}")

    # Test greedy generation strategy
    print("\nGenerated sequences using greedy strategy:")
    for i in range(3):
        generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'],
                                 sample_method='argmax')
        generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
        print(f"Sequence {i+1}: {generated_text}")

    '''
    Train standard model
    '''
    # Train model using standard training approach
    model, loss_hist, perplexity_hist = train(model, dataloader, 12, criterion,
                                             lr=1e-3, print_every=1, clip=1)

    '''
    Visualize standard training results
    '''
    plot_metrics(loss_hist, perplexity_hist, "Standard ")

    '''
    Initialize and train model using TBPTT
    '''
    # Create new model with larger hidden size for TBPTT
    model_tbtt = TextModel(word_to_int, hidden_size=2048).to(DEVICE)
    model_tbtt, loss_hist_tbbtt, perplexity_hist_tbbtt = train_tbbtt(
        model_tbtt, dataloader, 5, criterion,
        chunk_len=35, lr=1e-3,
        print_every=1, clip=1
    )

    '''
    Visualize TBPTT training results
    '''
    plot_metrics(loss_hist_tbbtt, perplexity_hist_tbbtt, "TBPTT ")

    '''
    Evaluate Standard model performance
    '''
    print("\nInitial prompt:", prompt_text)

    # Test random sampling generation
    print("\nGenerated sequences using random sampling (topk=5):")
    for i in range(3):
        generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'],
                                 sample_method='random', topk=5)
        generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
        print(f"Sequence {i+1}: {generated_text}")

    # Test greedy generation strategy
    print("\nGenerated sequences using greedy strategy:")
    for i in range(3):
        generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'],
                                 sample_method='argmax')
        generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
        print(f"Sequence {i+1}: {generated_text}")


    '''
    Bonus question
    '''
    def test_embedding_analogy(model, word_to_int, int_to_word):
        """
        Tests if embeddings show properties similar to word2vec.
        """ 
        
        with torch.no_grad():
            # Get word vectors
            v_trump = model.embedding(torch.tensor([word_to_int["trump"]], device=DEVICE))
            v_republican = model.embedding(torch.tensor([word_to_int["republican"]], device=DEVICE))
            v_biden = model.embedding(torch.tensor([word_to_int["biden"]], device=DEVICE))
            
            # Calculate: republican - trump + biden
            result_vec = v_republican - v_trump + v_biden
            
            # Find closest words to result
            embedding_weights = model.embedding.weight
            distances = torch.norm(embedding_weights - result_vec, dim=1)
            closest_indices = torch.topk(distances, k=5, largest=False).indices
            
            print("\nTop 5 closest words to the result vector:")
            for idx in closest_indices:
                word = int_to_word[idx.item()]
                dist = distances[idx].item()
                if word not in ['<EOS>', 'PAD']:
                    print(f"{word}: {dist:.4f}")
                

    test_embedding_analogy(model, word_to_int, int_to_word)
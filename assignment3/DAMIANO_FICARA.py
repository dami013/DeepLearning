
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

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.benchmark = False

def keys_to_values(keys, map, default_if_missing=None):
    """Convert keys to their corresponding values in the mapping."""
    return [map.get(key, default_if_missing) for key in keys]

class NewsSDataset(Dataset):
    """Dataset class for news headlines."""
    def __init__(self, tokenized_sequences, word_to_int):
        self.sequences = []
        for seq in tokenized_sequences:
            int_seq = [word_to_int.get(word, word_to_int['<EOS>']) for word in seq]
            self.sequences.append(int_seq)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return (
            torch.tensor(sequence[:-1]),
            torch.tensor(sequence[1:])
        )

def collate_fn(batch, pad_value):
    """Pads sequences in batch to same length."""
    x_seqs, y_seqs = zip(*batch)
    x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=pad_value)
    y_padded = pad_sequence(y_seqs, batch_first=True, padding_value=pad_value)
    return x_padded, y_padded

class Model(nn.Module):
    """LSTM-based language model for text generation."""
    def __init__(self, map, hidden_size=1024, emb_dim=150, n_layers=1):
        super(Model, self).__init__()

        self.vocab_size = len(map)
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.emb_dim,
            padding_idx=map["PAD"])

        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True)

        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        yhat, state = self.lstm(embed, prev_state)
        out = self.fc(yhat)
        return out, state

    def init_state(self, b_size=1):
        return (torch.zeros(self.n_layers, b_size, self.hidden_size),
                torch.zeros(self.n_layers, b_size, self.hidden_size))

def random_sample_next(model, x, prev_state, topk=5, uniform=True):
    """Sample next word randomly from top-k predictions."""
    out, state = model(x.to(DEVICE), prev_state=tuple(s.to(DEVICE) for s in prev_state))
    last_out = out[0, -1, :]
    
    topk = topk if topk else last_out.shape[0]
    top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)
    
    p = None if uniform else F.softmax(top_logit, dim=-1).cpu().detach().numpy()
    
    sampled_ix = np.random.choice(top_ix.cpu().numpy(), p=p)
    
    return sampled_ix, state

def sample_argmax(model, x, prev_state):
    """Pick the most likely next word."""
    out, state = model(x.to(DEVICE), prev_state=tuple(s.to(DEVICE) for s in prev_state))
    last_pred = out[:, -1, :]
    next_word = torch.argmax(last_pred, dim=-1).item()
    
    return next_word, state

def sample(model, prompt, stop_on, sample_method='random', topk=5, max_seqlen=18):
    """Generate text sequence based on prompt."""
    prompt = prompt if isinstance(prompt, (list, tuple)) else [prompt]

    model.eval()
    with torch.no_grad():
        sampled_ix_list = prompt[:]
        x = torch.tensor([prompt])
        prev_state = model.init_state(b_size=1)
        
        for t in range(max_seqlen - len(prompt)):
            if sample_method == 'argmax':
                sampled_ix, prev_state = sample_argmax(model, x, prev_state)
            else:
                sampled_ix, prev_state = random_sample_next(model, x, prev_state, topk=topk)

            sampled_ix_list.append(sampled_ix)
            x = torch.tensor([[sampled_ix]])
            
            if sampled_ix == stop_on:
                break

    model.train()
    return sampled_ix_list

def train(model, data, num_epochs, criterion, lr=0.001, print_every=50, clip=None):
    """Standard training loop."""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    model.train()
    
    loss_hist = []
    generated_text_list = []
    perplexity_hist = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_batches = len(data)
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(data, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            
            prev_state = model.init_state(b_size=x.shape[0])
            prev_state = tuple(s.to(DEVICE) for s in prev_state)
            
            out, state = model(x, prev_state=prev_state)
            loss_out = out.permute(0, 2, 1)
            
            loss = criterion(loss_out, y)
            epoch_loss += loss.item()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / total_batches
        loss_hist.append(avg_epoch_loss)
        
        perplexity = torch.exp(torch.tensor(avg_epoch_loss))
        perplexity_hist.append(perplexity.item())
        
        if print_every and (epoch % print_every) == 0:
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:8.4f}, Perplexity: {perplexity:8.4f}")
            generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'], sample_method='argmax')
            generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
            generated_text_list.append(generated_text)
            print(f"Generated text: {generated_text}\n")
            
        if avg_epoch_loss < 1.5:
            print(f"\nTarget loss of 1.5 reached at epoch {epoch}!")
            break
        
    return model, loss_hist, perplexity_hist

def train_tbbtt(model, data, num_epochs, criterion, truncate_length=50, lr=0.001, print_every=50, clip=None):
    """Training with Truncated Backpropagation Through Time."""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    model.train()
    
    loss_hist = []
    generated_text_list = []
    perplexity_hist = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_batches = len(data)
    epoch = 0
    
    while epoch < num_epochs:
        epoch += 1
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(data, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            seq_length = x.size(1)
            
            current_state = model.init_state(b_size=x.shape[0])
            current_state = tuple(s.to(DEVICE) for s in current_state)
            
            chunk_losses = []
            for chunk_start in range(0, seq_length, truncate_length):
                optimizer.zero_grad()
                
                chunk_end = min(chunk_start + truncate_length, seq_length)
                
                chunk_x = x[:, chunk_start:chunk_end]
                chunk_y = y[:, chunk_start:chunk_end]
                
                chunk_out, new_state = model(chunk_x, prev_state=current_state)
                
                loss_out = chunk_out.permute(0, 2, 1)
                
                loss = criterion(loss_out, chunk_y)
                chunk_losses.append(loss.item())
                
                loss.backward()
                
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                
                optimizer.step()
                
                current_state = tuple(s.detach() for s in new_state)
            
            batch_loss = sum(chunk_losses) / len(chunk_losses)
            epoch_loss += batch_loss
        
        avg_epoch_loss = epoch_loss / total_batches
        loss_hist.append(avg_epoch_loss)
        
        perplexity = torch.exp(torch.tensor(avg_epoch_loss))
        perplexity_hist.append(perplexity.item())
        
        if print_every and (epoch % print_every) == 0:
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:8.4f}, Perplexity: {perplexity:8.4f}")
            generated_indices = sample(model, prompt_indices, word_to_int['<EOS>'], sample_method='argmax')
            generated_text = " ".join(keys_to_values(generated_indices, int_to_word))
            generated_text_list.append(generated_text)
            print(f"Generated text: {generated_text}\n")
        
        if avg_epoch_loss < 1.5:
            print(f"\nTarget loss of 1.5 reached at epoch {epoch}!")
            break
    
    return model, loss_hist, perplexity_hist

def plot_metrics(loss_hist, perplexity_hist, title_prefix=""):
    """Plot training metrics."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist, label='Training Loss')
    plt.axhline(y=1.5, color='r', linestyle='--', label='Target Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix}Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(perplexity_hist, label='Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(f'{title_prefix}Perplexity Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' 
        if torch.backends.mps.is_available() else 'cpu')
    
    # Load dataset
    ds = load_dataset("heegyu/news-category-dataset")
    ds = ds['train'].filter(lambda x: x['category'] == 'POLITICS')
    
    # Tokenization
    tokenized_text = []
    for title in ds['headline']:
        words = title.lower().split()
        words.append('<EOS>')
        tokenized_text.append(words)
    
    # Create word mappings
    word_freq = {}
    for sequence in tokenized_text:
        for word in sequence:
            word_freq[word] = word_freq.get(word, 0) + 1
            
    word_to_int = {'<EOS>': 0}
    current_idx = 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for word, freq in sorted_words:
        if word != '<EOS>':
            word_to_int[word] = current_idx
            current_idx += 1
    
    word_to_int['PAD'] = len(word_to_int)
    int_to_word = {i: word for word, i in word_to_int.items()}
    
    # Create dataset and dataloader
    batch_size = 32
    dataset = NewsSDataset(tokenized_text, word_to_int)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                          collate_fn=lambda b: collate_fn(b, word_to_int["PAD"]),
                          shuffle=True)
    
    # Train standard model
    model = Model(word_to_int).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])
    
    prompt_text = "the president wants"
    prompt_indices = keys_to_values(prompt_text.split(), word_to_int)
    
    model, loss_hist, perplexity_hist = train(model, dataloader, 12, criterion, 
                                            lr=1e-3, print_every=1, clip=1)
    
    plot_metrics(loss_hist, perplexity_hist, "Standard ")
    
    # Train TBBTT model
    model2 = Model(word_to_int, hidden_size=2048).to(DEVICE)
    model2, loss_hist_tbbtt, perplexity_hist_tbbtt = train_tbbtt(
        model2, dataloader, 5, criterion,
        truncate_length=25, lr=1e-3,
        print_every=1, clip=1
    )
    
    plot_metrics(loss_hist_tbbtt, perplexity_hist_tbbtt, "TBBTT ")

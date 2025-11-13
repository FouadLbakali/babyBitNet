import torch
from gpt import GPTConfig, GPT
from bitnet import BitNetConfig, BitNet, BitLinear

# data loading
def get_batch(split, train_data, val_data, config):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])
    x, y = x.to(config["device"]), y.to(config["device"])
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    if config["model_type"] == "bitnet":
        for module in model.modules():
            if isinstance(module, BitLinear):
                module.quantize_for_inference()
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split, train_data, val_data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(
        model_type,
        batch_size = 16, # how many independent sequences will we process in parallel?
        block_size = 32, # what is the maximum context length for predictions?
        vocab_size = 65,
        max_iters = 5000,
        eval_interval = 100,
        learning_rate = 1e-3,
        eval_iters = 200,
        n_embd = 64,
        n_head = 4,
        n_layer = 4,
        dropout = 0.0,
        seed = None,
        two_stages = True,
        ):
    if seed:
        torch.manual_seed(seed) # default seed : 1337
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dict(
        model_type=model_type,
        batch_size=batch_size,
        block_size=block_size, 
        max_iters=max_iters, 
        eval_interval=eval_interval, 
        learning_rate=learning_rate, 
        eval_iters=eval_iters, 
        n_embd=n_embd, 
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device
        )
    model_args = dict(
        block_size=block_size, 
        vocab_size=vocab_size,
        n_embd=n_embd, 
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )
    if model_type == "gpt":
        model_config = GPTConfig(**model_args)
        model = GPT(model_config)
    elif model_type == "bitnet":
        model_config = BitNetConfig(**model_args)
        model = BitNet(model_config)
    else:
        return print("ERROR please choose one model between 'bitnet' or 'gpt'")
    model = model.to(device)
    # model.train()
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    train_losses = []
    val_losses = []
    steps = []

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            steps.append(iter)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # if model_type == "bitnet" and iter == max_iters // 2 and two_stages:
        #     # We reduce learning rate at the middle of training if the model is bitnet
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=(learning_rate/1.5))


        # sample a batch of data
        xb, yb = get_batch("train", train_data, val_data, config)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model, steps, train_losses, val_losses


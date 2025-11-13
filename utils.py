import pandas as pd
import torch
from bitnet import BitLinear

def save_train_loss(columns_name, train_loss):
    """Save training loss in a csv file"""
    csv_file_path = "results.csv"
    df = pd.read_csv(csv_file_path)
    # Convert a list of PyTorch tensors to a list of floats
    results = [tensor.item() for tensor in train_loss]

    df[columns_name] = results
    df.to_csv(csv_file_path, index=False)

def init_bitnet(model):
    """Prepare bitnet model for final inference"""
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.quantize_for_inference(cleenup=True)

def print_model_size(model, model_type=None): # Not used
    """Calculates and print the model size in Megabytes (MB)."""
    if model_type == "bitnet":
        init_bitnet(model)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Model size: {size_all_mb:.3f} MB')

def print_usage_vram(model, model_type=None): # Not used
    """Calculates and print the usage VRAM of model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    if model_type == "bitnet":
        init_bitnet(model)
    model.eval()

    print(f"VRAM allocated after model loading:: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
        
    peak_memory = torch.cuda.max_memory_allocated(device)

    print(f"\nPeak VRAM allocated during generation: {peak_memory / 1e6:.2f} MB")
    print(f"VRAM allocated after the end of generation: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    # create a mapping from integers to characters
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    print("\n--- Generated Text ---")
    print(decode(generated_tokens))
    print("--------------------")
import torch as tr

def get_device(choice=None):
    if choice == "cuda" and tr.cuda.is_available():
        print("cuda is available")
        return tr.device("cuda")
    if choice == "mps" and tr.mps.is_available():
        print("mps is available")
        return tr.device("mps")
    # fallback to first available
    if tr.cuda.is_available():
        print("cuda is available")
        return tr.device("cuda")
    if tr.mps.is_available():
        print("mps is available")
        return tr.device("mps")
    print("Using CPU")
    return tr.device("cpu")

import torch

success = torch.cuda.is_available()

print(f"{'Success! cuda available.' if success else 'uh oh, no cuda :('}")

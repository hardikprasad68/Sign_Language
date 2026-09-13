import os
import sys
import torch

# 1. Add 'src' to the Python path so we can import your custom models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

from models.static_model import StaticModelV2  # type: ignore
from models.dynamic_model import DynamicGRU  # type: ignore

def export_models():
    print("Reading class counts from label maps...")
    # Count the number of classes dynamically from your saved text files
    with open(os.path.join(project_root, 'models', 'class_labels.txt'), 'r') as f:
        static_classes = len(f.read().splitlines())
    with open(os.path.join(project_root, 'models', 'dynamic_labels.txt'), 'r') as f:
        dynamic_classes = len(f.read().splitlines())

    print("Initializing empty model architectures...")
    # 2. Instantiate the empty models using the parameters from train.py & train_dynamic.py
    static_model = StaticModelV2(input_dim=126, num_classes=static_classes, hidden_dim=256, dropout_prob=0.25)
    dynamic_model = DynamicGRU(input_dim=126, hidden_dim=128, num_layers=2, num_classes=dynamic_classes)

    print("Loading weights into models...")
    # 3. Load the OrderedDict weights into the architectures
    static_model.load_state_dict(torch.load(os.path.join(project_root, 'models', 'mlp_v2.pt')))
    dynamic_model.load_state_dict(torch.load(os.path.join(project_root, 'models', 'dynamic_v1.pt')))
    
    # 4. Set to evaluation mode
    static_model.eval()
    dynamic_model.eval()
    
    print("Tracing architectures...")
    # 5. Create dummy tensors (DynamicGRU expects 30 frames as defined in your resampling logic)
    static_dummy_input = torch.randn(1, 126)
    dynamic_dummy_input = torch.randn(1, 30, 126)
    
    static_traced = torch.jit.trace(static_model, static_dummy_input)
    dynamic_traced = torch.jit.trace(dynamic_model, dynamic_dummy_input)
    
    print("Saving TorchScript files...")
    # 6. Save the fused models
    static_traced.save(os.path.join(project_root, 'models', 'static_traced.pt'))
    dynamic_traced.save(os.path.join(project_root, 'models', 'dynamic_traced.pt'))
    
    print("\n[SUCCESS] Models exported successfully! Person 3 can now load them directly.")

if __name__ == "__main__":
    export_models()
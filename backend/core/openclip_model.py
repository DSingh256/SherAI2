"""
VanRakshak AI - Shared OpenCLIP Model Singleton
Provides a unified model instance for zero-shot classification and semantic verification.
"""

import torch

class SharedOpenCLIP:
    _model = None
    _preprocess = None
    _tokenizer = None
    _device = None

    @classmethod
    def get_device(cls) -> str:
        if cls._device is None:
            if torch.cuda.is_available():
                cls._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                cls._device = "mps"
            else:
                cls._device = "cpu"
        return cls._device

    @classmethod
    def get_model(cls):
        if cls._model is None:
            try:
                import open_clip
                
                device = cls.get_device()
                print(f"Loading OpenCLIP ViT-B-32 on {device}...")
                
                # Load ViT-B-32 trained on LAION-2B
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', 
                    pretrained='laion2b_s34b_b79k',
                    device=device
                )
                
                # Enable FP16 (half precision) for efficiency on GPU/MPS
                if device in ["cuda", "mps"]:
                    model = model.half()
                
                model.eval()
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
                
                cls._model = model
                cls._preprocess = preprocess
                cls._tokenizer = tokenizer
                print("OpenCLIP model loaded successfully.")
                
            except ImportError:
                print("open_clip_torch not installed.")
                return None, None, None
            except Exception as e:
                print(f"Error loading OpenCLIP: {e}")
                return None, None, None
                
        return cls._model, cls._preprocess, cls._tokenizer

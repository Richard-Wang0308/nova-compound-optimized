"""
Optimized Boltz-2 model loading and management.

Integrates with existing BoltzWrapper for model caching.
"""

import os
import time
from typing import Dict, Optional, Any
from pathlib import Path
from threading import Lock  # FIX: Use threading.Lock instead
import torch
import gc
import bittensor as bt


# ============================================================================
# RESIDENT MODEL MANAGER (Keep Models in GPU Memory)
# ============================================================================

class ResidentModelManager:
    """
    Manages models that stay resident in GPU memory.
    
    Compatible with BoltzWrapper's model caching strategy.
    Thread-safe implementation.
    """
    
    def __init__(self):
        self.models = {}  # {(model_name, device_id): model}
        self.load_times = {}
        self.last_used = {}
        self.lock = Lock()  # FIX: Always use threading.Lock
        
        bt.logging.info("ResidentModelManager initialized - models will stay in GPU memory")
    
    def load_model(
        self,
        model_name: str,
        model_loader_func,  # Function that returns loaded model
        device_id: int = 0,
        force_reload: bool = False
    ) -> torch.nn.Module:
        """
        Load model and keep it resident in GPU memory.
        Thread-safe implementation.
        """
        key = (model_name, device_id)
        
        # FIX: Check if already loaded (with lock)
        with self.lock:
            if key in self.models and not force_reload:
                bt.logging.debug(f"Model {model_name} already resident on GPU {device_id}")
                self.last_used[key] = time.time()
                return self.models[key]
        
        # Load model (outside lock to allow parallel loading of different models)
        bt.logging.info(f"Loading model {model_name} to GPU {device_id}...")
        start_time = time.time()
        
        try:
            # Set CUDA device before loading
            if torch.cuda.is_available():
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load model using provided function
            model = model_loader_func()
            
            # Ensure model is on correct device
            if torch.cuda.is_available():
                model = model.to(f"cuda:{device_id}")
            
            model.eval()
            
            # FIX: Store in cache (with lock)
            load_time = time.time() - start_time
            with self.lock:
                self.models[key] = model
                self.load_times[key] = load_time
                self.last_used[key] = time.time()
            
            bt.logging.info(
                f"Model {model_name} loaded to GPU {device_id} in {load_time:.2f}s "
                f"(will stay resident)"
            )
            
            return model
        
        except Exception as e:
            bt.logging.error(f"Failed to load model {model_name}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            raise
    
    def get_model(self, model_name: str, device_id: int = 0) -> Optional[torch.nn.Module]:
        """Get a resident model if it exists (thread-safe)."""
        with self.lock:  # FIX: Add lock
            key = (model_name, device_id)
            if key in self.models:
                self.last_used[key] = time.time()
                return self.models[key]
            return None
    
    def unload_model(self, model_name: str, device_id: int = 0):
        """Explicitly unload a model from GPU memory (thread-safe)."""
        key = (model_name, device_id)
        
        # FIX: Check existence with lock
        with self.lock:
            if key not in self.models:
                return
            model = self.models[key]
        
        # Do expensive operations outside lock
        model.cpu()
        
        # FIX: Delete with lock (double-check pattern)
        with self.lock:
            if key in self.models:
                del self.models[key]
        
        del model
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_id)
        
        bt.logging.info(f"Unloaded model {model_name} from GPU {device_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models (thread-safe)."""
        # FIX: Create snapshot with lock
        with self.lock:
            models_snapshot = dict(self.models)
            load_times_snapshot = dict(self.load_times)
            last_used_snapshot = dict(self.last_used)
        
        # Process snapshot outside lock
        stats = {
            'num_models_loaded': len(models_snapshot),
            'models': {}
        }
        
        for (model_name, device_id), model in models_snapshot.items():
            # Calculate model size
            try:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                total_size_mb = (param_size + buffer_size) / 1024 / 1024
            except:
                total_size_mb = 0
            
            stats['models'][f"{model_name}_gpu{device_id}"] = {
                'device_id': device_id,
                'size_mb': total_size_mb,
                'load_time': load_times_snapshot.get((model_name, device_id), 0),
                'last_used': last_used_snapshot.get((model_name, device_id), 0)
            }
        
        return stats


# ============================================================================
# torch.compile() WRAPPER (PyTorch 2.0+)
# ============================================================================

class CompiledModelWrapper:
    """
    Wrapper for torch.compile() optimization.
    
    Note: torch.compile() may not work well with PyTorch Lightning.
    Use with caution.
    """
    
    def __init__(self, enable_compile: bool = False):  # Default False for safety
        self.enable_compile = enable_compile
        self.compiled_models = {}
        
        # Check PyTorch version
        try:
            import packaging.version
            torch_version = packaging.version.parse(torch.__version__.split('+')[0])
            required_version = packaging.version.parse("2.0.0")
            self.compile_available = torch_version >= required_version
        except:
            torch_version_str = torch.__version__.split('+')[0]
            major, minor = map(int, torch_version_str.split('.')[:2])
            self.compile_available = (major >= 2)
        
        if self.enable_compile and not self.compile_available:
            bt.logging.warning(
                f"torch.compile() requires PyTorch 2.0+, current: {torch.__version__}. "
                "Compilation disabled."
            )
            self.enable_compile = False
        
        if self.enable_compile:
            bt.logging.info("torch.compile() enabled - models will be JIT compiled")
            bt.logging.warning("Note: torch.compile() may not work well with PyTorch Lightning")
    
    def compile_model(
        self,
        model: torch.nn.Module,
        mode: str = "default",
        dynamic: bool = False
    ) -> torch.nn.Module:
        """
        Compile model with torch.compile().
        """
        if not self.enable_compile:
            return model
        
        model_id = id(model)
        if model_id in self.compiled_models:
            bt.logging.debug("Model already compiled")
            return self.compiled_models[model_id]
        
        try:
            bt.logging.info(f"Compiling model with torch.compile(mode={mode})...")
            start_time = time.time()
            
            compiled_model = torch.compile(
                model,
                mode=mode,
                dynamic=dynamic,
                backend="inductor"
            )
            
            compile_time = time.time() - start_time
            bt.logging.info(f"Model compiled in {compile_time:.2f}s")
            
            self.compiled_models[model_id] = compiled_model
            return compiled_model
        
        except Exception as e:
            bt.logging.warning(f"torch.compile() failed: {e}. Using uncompiled model.")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return model
    
    def warm_up_model(self, model: torch.nn.Module, dummy_input: Any, num_warmup: int = 3):
        """Warm up compiled model with dummy inputs."""
        if not self.enable_compile:
            return
        
        bt.logging.info(f"Warming up compiled model ({num_warmup} iterations)...")
        model.eval()
        
        success_count = 0
        with torch.no_grad():
            for i in range(num_warmup):
                try:
                    _ = model(dummy_input)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    success_count += 1
                except Exception as e:
                    bt.logging.error(f"Warmup iteration {i} failed: {e}")
                    import traceback
                    bt.logging.debug(traceback.format_exc())
        
        if success_count == 0:
            raise RuntimeError("Model warmup completely failed")
        elif success_count < num_warmup:
            bt.logging.warning(f"Only {success_count}/{num_warmup} warmup iterations succeeded")
        else:
            bt.logging.info("Model warmup complete")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_model_manager() -> ResidentModelManager:
    """
    Create a resident model manager.
    
    Returns:
        ResidentModelManager instance
    """
    return ResidentModelManager()

from collections import defaultdict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerCallback
)
from helpers.logging import logger

class TrainingMonitor:
    """Handles training monitoring and metrics logging."""
    
    @staticmethod
    def memory_stats() -> None:
        """Log current GPU memory statistics."""
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU Memory - Allocated: {allocated:.2f} GB, "
            f"Reserved: {reserved:.2f} GB"
        )

class MemoryStatsCallback(TrainerCallback):
    """Callback to log memory usage during training."""
    
    def on_step_end(self, args, state, control, **kwargs):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"[Step {state.global_step}] "
            f"Memory - Allocated: {allocated:.2f} GB, "
            f"Reserved: {reserved:.2f} GB"
        )

class LossLoggingCallback(TrainerCallback):
    """Callback to log training loss."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                logger.info(
                    f"[Step {state.global_step}] Loss: {last_log['loss']:.4f}"
                )
            else:
                logger.debug(f"[Step {state.global_step}] No loss logged")

class WeightsLoggingCallback(TrainerCallback):
    def __init__(self, param_name_substring="weight", log_every_n_steps=10):
        self.param_name_substring = param_name_substring
        self.log_every_n_steps = log_every_n_steps
        self.logged_weights = []
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Verify we're entering training steps"""
        # if state.global_step % self.log_every_n_steps == 0:
        #     print(f"\n[WeightsLogging] Step {state.global_step} begin")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return

        model = kwargs.get("model")
        if model is None:
            print("Warning: No model in kwargs!")
            return

        # #print(f"\n[WeightsLogging] Recording at step {state.global_step}")
        for name, param in model.named_parameters():
            if self.param_name_substring in name and param.requires_grad:
                # print(f"  {name}: requires_grad={param.requires_grad}, grad={param.grad is not None}")
                self.logged_weights.append((
                    state.global_step, 
                    name, 
                    param.detach().cpu().clone()
                ))

# TODO: may need to rewrite training_step in trainer.py to get grad information.
class GradientLoggingCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self.step_gradients = defaultdict(list)
        
    # def on_backward_end(self, args, state, control, **kwargs):
    #     print(f"\n[GradientLogging] Backward end at step {state.global_step}")
        
    #     model = kwargs.get("model")
    #     if model is None:
    #         print("Warning: No model in kwargs!")
    #         return

    #     any_grads = False
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             any_grads = True
    #             avg_grad = param.grad.abs().mean().item()
    #             self.step_gradients[name].append((state.global_step, avg_grad))
    #             print(f"  {name}: grad_norm={avg_grad:.4e}")

    #     if not any_grads:
    #         print("  No gradients found for any parameters!")

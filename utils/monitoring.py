import torch
from transformers import TrainerCallback
from utils.logging import logger


class TrainingMonitor:
    """Handles training monitoring and metrics logging."""
    @staticmethod
    def memory_stats() -> None:
        """Log current GPU memory statistics."""
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, "
                    f"Reserved: {reserved:.2f} GB")


class MemoryStatsCallback(TrainerCallback):
    """Callback to log memory usage during training."""
    def on_step_end(self, args, state, control, **kwargs):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[Step {state.global_step}] "
                    f"Memory - Allocated: {allocated:.2f} GB, "
                    f"Reserved: {reserved:.2f} GB")


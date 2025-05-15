import torch
from transformers import TrainerCallback
from libs.utils.logging_utils import logger


class TrainingMonitor:
    """Handles training monitoring and metrics logging."""
    @staticmethod
    def memory_stats() -> None:
        """Log current GPU memory statistics."""
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, "
                    f"Reserved: {reserved:.2f} GB")


class GPUMemoryStatsCallback(TrainerCallback):
    """Callback to log memory usage during training."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[Step {state.global_step}] "
                        f"Memory - Allocated: {allocated:.2f} GB, "
                        f"Reserved: {reserved:.2f} GB")

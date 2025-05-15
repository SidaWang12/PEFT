import logging
from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers.trainer_callback import TrainerState

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_training_metrics(trainer_state: TrainerState,
                         sum_training_step_time: float,
                         flops_profiler: FlopsProfiler) -> None:
    train_losses = [
        log["loss"] for log in trainer_state.log_history if "loss" in log
    ]
    eval_losses = [
        log["eval_loss"] for log in trainer_state.log_history
        if "eval_loss" in log
    ]
    logger.info(f"train_losses: {train_losses}")
    logger.info(f"eval_losses: {eval_losses}")

    logger.info(
        f"Average training step  {sum_training_step_time / trainer_state.global_step}s"
    )

    logger.info(f"FLOPs: {flops_profiler.get_total_flops() / 1e12} TFLOPS")

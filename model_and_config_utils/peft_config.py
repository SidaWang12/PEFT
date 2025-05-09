from dataclasses import dataclass, field
from trl import SFTConfig


@dataclass
class PeftConfig(SFTConfig):
    """
    PEFT (Parameter-Efficient Fine-Tuning) specific configuration.
    """
    enable_analysis: bool = field(
        default=False,
        metadata={"help": "whether or not do gradient analysis."},
    )
    downsample_attention_blocks_ratio: float = field(
        default=0.005,
        metadata={"help": "downsample_attention_blocks_ratio."},
    )
    test_set_percentage: float = field(
        default=0.2,
        metadata={"help": "test set percentage."},
    )

from dataclasses import dataclass, field
from trl import SFTConfig


@dataclass
class SMTConfig(SFTConfig):
    """
    PEFT (Parameter-Efficient Fine-Tuning) specific configuration.
    """
    enable_analysis: bool = field(
        default=False,
        metadata={"help": "whether or not do gradient analysis."},
    )
    downsample_mlp_blocks_ratio: float = field(
        default=-1.0,
        metadata={
            "help":
            'Proportion of selected mlp blocks'
            'relative to the total number of all blocks '
            '(not just mlp blocks).Set to negative to turn it off.'
        },
    )
    downsample_attention_blocks_ratio: float = field(
        default=-1.0,
        metadata={
            "help":
            'Proportion of selected attention blocks'
            'relative to the total number of all blocks '
            '(not just attention blocks).Set to negative to turn it off.'
        },
    )
    test_set_percentage: float = field(
        default=0.2,
        metadata={"help": "test set percentage."},
    )

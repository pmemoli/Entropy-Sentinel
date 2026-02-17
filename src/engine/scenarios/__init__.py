from .livemathbench import LIVEMATHBENCH
from .gsmsymbolic import GSM8KSYMBOLIC
from .gsm import GSM8K
from .mathhendrycks import MATH
from .gpqa import GPQA
from .scibench import SCIBENCH
from .svamp import SVAMP
from .theoremqa import THEOREMQA
from .olympiadbench import OLYMPIADBENCH
from .matscibench import MATSCIBENCH

REGISTRY = {
    "gsm8k": GSM8K,
    "gsm8ksymbolic": GSM8KSYMBOLIC,
    "mathhendrycks": MATH,
    "gpqa": GPQA,
    "scibench": SCIBENCH,
    "svamp": SVAMP,
    "livemathbench": LIVEMATHBENCH,
    "theoremqa": THEOREMQA,
    "olympiadbench": OLYMPIADBENCH,
    "matscibench": MATSCIBENCH,
}

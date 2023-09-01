from .inference_core import XMem, log
from .memory_manager import MemoryManager
from .kv_memory_store import KeyValueMemoryStore
from .track import Track
# from ..model.network import XMem as XMemModel
# from ..checkpoint import ensure_checkpoint
# from .config import DEFAULT_CONFIG

# class XMem(XMemModel):
#     def __init__(self, config=None, model_path=None, map_location=None):
#         self.config = config = config or DEFAULT_CONFIG
#         model_path = model_path or ensure_checkpoint()
#         super().__init__(config, model_path, map_location)
#         self.eval()

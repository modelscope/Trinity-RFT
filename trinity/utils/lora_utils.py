from verl.utils.vllm_utils import VLLMHijack as VLLMHijackBase


class VLLMHijack(VLLMHijackBase):
    _hijacked = False

    @classmethod
    def hijack(cls):
        if cls._hijacked:
            return

        super().hijack()
        
        cls._hijacked = True
        print("!!! [vLLM Patch] Tensor-based LoRA loading enabled.")

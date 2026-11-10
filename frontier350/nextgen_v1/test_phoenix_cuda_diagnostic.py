"""CPU-only synthetic guard checks; never execute CUDA or load model weights."""
import types
import unittest

from phoenix_cuda_diagnostic import MIN_FREE_MIB, require_cuda_ready


class FakeCuda:
    def __init__(self, available=True, bf16=True, count=1,
                 name="NVIDIA GeForce RTX 4070", free_mib=10000):
        self.available, self.bf16, self.count = available, bf16, count
        self.name, self.free_mib = name, free_mib

    def is_available(self):
        return self.available

    def is_bf16_supported(self):
        return self.bf16

    def device_count(self):
        return self.count

    def get_device_name(self, index):
        assert index == 0
        return self.name

    def mem_get_info(self, index):
        assert index == 0
        return self.free_mib * 1024 * 1024, 12 * 1024 * 1024 * 1024


class CudaDiagnosticContract(unittest.TestCase):
    def test_idle_known_card_passes_readiness(self):
        self.assertIsNone(require_cuda_ready(types.SimpleNamespace(cuda=FakeCuda())))

    def test_missing_cuda_or_bf16_refuses(self):
        for cuda in (FakeCuda(available=False), FakeCuda(bf16=False)):
            with self.subTest(cuda=cuda.__dict__), self.assertRaises(RuntimeError):
                require_cuda_ready(types.SimpleNamespace(cuda=cuda))

    def test_wrong_card_or_multiple_cards_refuse(self):
        for cuda in (FakeCuda(name="A100"), FakeCuda(count=2)):
            with self.subTest(cuda=cuda.__dict__), self.assertRaises(RuntimeError):
                require_cuda_ready(types.SimpleNamespace(cuda=cuda))

    def test_insufficient_vram_refuses(self):
        with self.assertRaises(RuntimeError):
            require_cuda_ready(types.SimpleNamespace(cuda=FakeCuda(free_mib=MIN_FREE_MIB - 1)))

    def test_vram_boundary_is_explicit(self):
        self.assertIsNone(require_cuda_ready(
            types.SimpleNamespace(cuda=FakeCuda(free_mib=MIN_FREE_MIB))))


if __name__ == "__main__":
    unittest.main()

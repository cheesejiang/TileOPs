"""Source-only binding checks without Triton, vLLM or an NPU."""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('vendor_source_tested', ROOT / 'benchmarks/ascend/vendor_source.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class VendorSourceTest(unittest.TestCase):
    def test_only_imports_change_in_native_functions(self):
        text = 'from vllm.triton_utils import triton as tr\n@tr.jit\ndef kernel(x):\n    return x + 1\n'
        original = ast.parse(text)
        rewritten = loader.rewrite_imports(text, 'vendor.py', {'triton': object()}, '_bindings')
        self.assertEqual(ast.dump(original.body[1]), ast.dump(rewritten.body[1]))
        self.assertEqual(rewritten.body[0].module, '_bindings')
        self.assertEqual(rewritten.body[0].names[0].asname, 'tr')

    def test_unrecognized_framework_dependency_fails(self):
        with self.assertRaises(RuntimeError):
            loader.rewrite_imports('from vllm.other import foo', 'bad.py', {}, '_bindings')
        with self.assertRaises(RuntimeError):
            loader.rewrite_imports('from vllm.triton_utils import unknown', 'bad.py', {}, '_bindings')

    def test_load_does_not_import_vendor_package_and_preserves_readable_source(self):
        class FakeTriton:
            @staticmethod
            def jit(fn):
                return fn
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.py'
            path.write_text('from vllm.triton_utils import triton\n@triton.jit\ndef kernel(x):\n    return x + 1\n')
            before = {name for name in sys.modules if name.startswith(('vllm', 'vllm_ascend'))}
            module, evidence = loader.load_source(path, {'triton': FakeTriton()})
            self.assertEqual(module.kernel(3), 4)
            self.assertIn('return x + 1', inspect.getsource(module.kernel))
            self.assertEqual(before, {name for name in sys.modules if name.startswith(('vllm', 'vllm_ascend'))})
            self.assertEqual(len(evidence['source_sha256']), 64)
            self.assertTrue(Path(evidence['generated_source']).is_file())


if __name__ == '__main__':
    unittest.main()

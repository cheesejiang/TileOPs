"""CPU-only routing checks; never import NPU or vendor runtimes."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class FinalBaselineRouting(unittest.TestCase):
    def factory(self):
        path = Path(__file__).resolve().parents[2] / 'benchmarks/ascend/gather_final_baselines.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_implementations')
        class Run:
            def __init__(self, inputs, **kwargs):
                self.inputs, self.options = inputs, kwargs
            def reset(self):
                pass
        namespace = {'torch': SimpleNamespace(float8_e4m3fn='fp8', float16='fp16', bfloat16='bf16'),
                     '_VllmFP8Run': Run, 'Implementation': lambda run, **kwargs: run}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        fake = ModuleType('benchmarks.ascend.gather_v2_baseline')
        fake._Run = Run
        return namespace['make_implementations'], fake

    def test_all_seven_have_exactly_one_baseline(self):
        factory, fake = self.factory()
        with patch.dict(sys.modules, {'benchmarks.ascend.gather_v2_baseline': fake}):
            for dtype in ('fp8', 'bf16', 'bf16', 'fp16', 'fp8', 'bf16', 'bf16'):
                case = SimpleNamespace(inputs=(None, SimpleNamespace(dtype=dtype)))
                result = factory('PagedKVCacheGatherFwdOp', case)
                expected = 'vllm-ascend-fp8-full' if dtype == 'fp8' else 'ops-nn-gatherv2-full'
                self.assertEqual(set(result), {expected})
                run = result[expected]
                self.assertIs(run.inputs, case.inputs)
                if dtype != 'fp8':
                    self.assertTrue(run.options['include_index_generation'])

    def test_other_ops_are_ignored(self):
        factory, _ = self.factory()
        self.assertEqual(factory('OtherOp', None), {})


if __name__ == '__main__':
    unittest.main()

"""CPU checks for DSL scopes that Python byte compilation cannot validate."""
import ast
from pathlib import Path
import unittest


class PrecomputeScopeTest(unittest.TestCase):
    def test_dma_uses_segment_variables_inside_segment_loop(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / 'src/tileops/kernels/ascend/attention/paged_cache_gather_precompute.py').read_text())
        main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loops = [n for n in ast.walk(main) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == 'segment']
        self.assertEqual(len(loops), 1)
        body = ast.Module(body=loops[0].body, type_ignores=[])
        stores = {n.id for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.assertTrue({'target', 'physical', 'count', 'contiguous'} <= stores)
        copies = [n for n in ast.walk(body) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'copy']
        self.assertEqual(len(copies), 2)
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'segments' for n in ast.walk(main)))
        # One static groups definition outside the active-core guard, reused at drain.
        definitions = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'groups' for t in n.targets)]
        self.assertEqual(len(definitions), 1)


if __name__ == '__main__':
    unittest.main()

"""Source-bound vendor loader: rebind approved framework imports only.

Standard-library imports only. No vendor package __init__ or NPU initialization
runs until the caller explicitly loads a source module under its device guard.
"""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import types


_ALLOWED = {'vllm.triton_utils', 'vllm_ascend', 'vllm_ascend.ops.triton.triton_utils'}


def rewrite_imports(source, filename, bindings, binding_module):
    tree = ast.parse(source, filename)
    class Rewrite(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            module = node.module or ''
            if module in _ALLOWED:
                missing = {alias.name for alias in node.names} - set(bindings)
                if missing:
                    raise RuntimeError(f'Unsupported vendor bindings in {filename}: {module}: {sorted(missing)}')
                node.module = binding_module
            elif module.split('.')[0] in ('vllm', 'vllm_ascend') or node.level:
                raise RuntimeError(f'Unadapted vendor import in {filename}: {module}')
            return node

        def visit_Import(self, node):
            if any(alias.name.split('.')[0] in ('vllm', 'vllm_ascend') for alias in node.names):
                raise RuntimeError(f'Unadapted vendor package import in {filename}')
            return node
    return Rewrite().visit(tree)


def load_source(path, bindings):
    """Materialize transformed source so Triton/inspect can read its JIT bodies."""
    path = Path(path).resolve(strict=True)
    original = path.read_text(encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = hashlib.sha256((str(path) + digest).encode()).hexdigest()[:20]
    name = '_tileops_vendor_' + path.stem + '_' + identity
    binding_name = name + '_bindings'
    tree = rewrite_imports(original, str(path), bindings, binding_name)
    generated = ast.unparse(tree) + '\n'
    folder = Path(tempfile.mkdtemp(prefix='tileops-vendor-'))
    target = folder / path.name
    target.write_text(generated, encoding='utf-8')
    injected = types.ModuleType(binding_name)
    injected.__dict__.update(bindings)
    module = types.ModuleType(name)
    module.__file__ = str(target)
    sys.modules[binding_name] = injected
    sys.modules[name] = module
    try:
        exec(compile(generated, str(target), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        sys.modules.pop(binding_name, None)
        raise
    evidence = {'source': str(path), 'source_sha256': digest,
                'generated_source': str(target),
                'generated_sha256': hashlib.sha256(generated.encode()).hexdigest(),
                'adaptation': 'approved-import-rebinding-only'}
    return module, evidence

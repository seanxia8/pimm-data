"""Architectural contract: the framework-neutral IO layer imports no torch.

pimm-data keeps torch a required *install* dependency (ADR §5), but the
reader / joint-index / decode layer must stay framework-neutral so a JAX or
plain-numpy consumer can read events (the readers return numpy) without the
torch transform/collate/Dataset layer. This is a static import-contract check
(import-linter's job in a few lines) and it runs in the normal suite, catching
torch creep at PR time. It does NOT assert a torch-free *import* of the package
(``import pimm_data`` is eager and pulls torch) — that is a deliberate non-goal.

If/when CI exists, this can migrate to a declarative ``import-linter`` contract.
"""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pimm_data"

# The transitive closure of the torch-free public surface (readers + the
# joint-index / shard-meta / label-decorate helpers they rely on).
TORCH_FREE_MODULES = [
    SRC / "_joint_index.py",
    SRC / "_shard_meta.py",
    SRC / "_label_decorate.py",
    *sorted((SRC / "readers").glob("*.py")),
]


def _imports_torch(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "torch" or a.name.startswith("torch.")
                   for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "torch" or mod.startswith("torch."):
                return True
    return False


@pytest.mark.parametrize(
    "path", TORCH_FREE_MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_io_layer_does_not_import_torch(path):
    assert path.exists(), f"expected torch-free module missing: {path}"
    assert not _imports_torch(path), (
        f"{path.relative_to(SRC)} imports torch — the reader/index/decode layer "
        f"must stay framework-neutral (ADR §5 / PR-E).")

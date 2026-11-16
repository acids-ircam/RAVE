"""Check fine-tuning beta selection and saved configuration without audio imports."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


class VariationalEncoder:
    def __init__(self, beta):
        self.beta = beta


def configure(model, beta=None):
    source = ast.parse((Path(__file__).parents[1] / 'rave/scripts/finetune.py').read_text())
    helper = next(node for node in source.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_configure_finetune_beta')
    bindings = {}
    scope = {
        'math': math,
        'rave': SimpleNamespace(blocks=SimpleNamespace(VariationalEncoder=VariationalEncoder)),
        'gin': SimpleNamespace(unlock_config=nullcontext,
                               bind_parameter=lambda name, value: bindings.update({name: value})),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
                 'rave/scripts/finetune.py', 'exec'), scope)
    scope['_configure_finetune_beta'](model, beta)
    return bindings


@pytest.mark.parametrize('source,beta,expected', [
    (0.5, None, 0.125),
    (0., None, 0.),
    (0.5, 0.01, 0.01),
    (0.5, 0., 0.),
])
def test_beta_selection_and_saved_binding(source, beta, expected):
    model = SimpleNamespace(encoder=VariationalEncoder(source))
    bindings = configure(model, beta)
    assert model.encoder.beta == expected
    assert bindings == {'blocks.VariationalEncoder.beta': expected}


@pytest.mark.parametrize('beta', [-1., float('nan'), float('inf'), -float('inf')])
def test_beta_rejects_invalid_values(beta):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        configure(SimpleNamespace(encoder=VariationalEncoder(0.5)), beta)


def test_nonvariational_regularization_is_unchanged():
    encoder = SimpleNamespace(regularization=0.5)
    assert configure(SimpleNamespace(encoder=encoder)) == {}
    assert encoder.regularization == 0.5
    with pytest.raises(ValueError, match='requires a variational encoder'):
        configure(SimpleNamespace(encoder=encoder), 0.01)

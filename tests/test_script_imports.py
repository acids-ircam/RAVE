import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["export", "preprocess", "adapt"])
def test_import_keeps_grad_enabled(module):
    # in a subprocess, as another test may have imported the module already
    code = f"import torch, rave.scripts.{module}; assert torch.is_grad_enabled()"
    subprocess.run([sys.executable, "-c", code], check=True)

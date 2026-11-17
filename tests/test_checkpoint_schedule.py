import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from rave.core import ModelCheckpoint


def test_schedules_and_resume(tmp_path):
    checkpoint = ModelCheckpoint(dirpath=tmp_path, step_period=5, last_period=2)
    saved = []
    checkpoint._save_checkpoint = lambda trainer, path: saved.append(Path(path).name)
    trainer = object()
    for index in range(5):
        checkpoint.on_train_batch_end(trainer, None, None, None, index)
    assert saved == ['last.ckpt', 'last.ckpt', 'step_5.ckpt']
    state = checkpoint.state_dict()
    resumed = ModelCheckpoint(dirpath=tmp_path, step_period=5, last_period=2)
    resumed.load_state_dict(state)
    resumed._save_checkpoint = lambda trainer, path: saved.append(Path(path).name)
    for index in range(5):
        resumed.on_train_batch_end(trainer, None, None, None, index)
    assert saved[3:] == ['last.ckpt', 'last.ckpt', 'step_10.ckpt', 'last.ckpt']
    resumed.on_train_end(trainer, None)
    resumed.on_exception(trainer, None, KeyboardInterrupt())
    assert saved[-2:] == ['last.ckpt', 'interrupt.ckpt']
    resumed.on_exception(trainer, None, RuntimeError('failure'))
    assert saved[-2:] == ['last.ckpt', 'interrupt.ckpt']


def test_lightning_saves_interrupt_checkpoint(tmp_path):
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def training_step(self, batch, batch_idx):
            if batch_idx == 3:
                raise KeyboardInterrupt()
            return self.weight.square()
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    callback = ModelCheckpoint(dirpath=tmp_path, step_period=200, last_period=100)
    loader = DataLoader(TensorDataset(torch.zeros(8, 1)), batch_size=1)
    trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False, callbacks=[callback],
                         enable_progress_bar=False, enable_model_summary=False, max_epochs=1)
    old_handler = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(SystemExit):
            trainer.fit(Model(), loader)
    finally:
        signal.signal(signal.SIGINT, old_handler)
    checkpoint = torch.load(tmp_path / 'interrupt.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == 3
    assert not list(tmp_path.glob('step_*.ckpt'))


def test_lightning_schedules_survive_resume(tmp_path):
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def training_step(self, batch, batch_idx):
            return self.weight.square()
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    loader = DataLoader(TensorDataset(torch.zeros(20, 1)), batch_size=1)
    def trainer(max_steps):
        cb = ModelCheckpoint(dirpath=tmp_path, step_period=4, last_period=2)
        return pl.Trainer(accelerator='cpu', devices=1, logger=False, callbacks=[cb],
                          enable_progress_bar=False, enable_model_summary=False, max_steps=max_steps)

    trainer(6).fit(Model(), loader)
    assert sorted(p.name for p in tmp_path.glob('*.ckpt')) == ['last.ckpt', 'step_4.ckpt']
    ckpt = torch.load(tmp_path / 'last.ckpt', map_location='cpu', weights_only=False)
    assert ckpt['global_step'] == 6
    trainer(9).fit(Model(), loader, ckpt_path=tmp_path / 'last.ckpt')
    assert sorted(p.name for p in tmp_path.glob('*.ckpt')) == ['last.ckpt', 'step_4.ckpt', 'step_8.ckpt']
    ckpt = torch.load(tmp_path / 'last.ckpt', map_location='cpu', weights_only=False)
    assert ckpt['global_step'] == 9

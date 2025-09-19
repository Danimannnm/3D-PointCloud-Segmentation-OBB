import os
import sys
import time
import torch

from train_Toronto3D import Toronto3DConfig
from datasets.Toronto3D import Toronto3DDataset, Toronto3DSampler, Toronto3DCollate
from torch.utils.data import DataLoader
from models.architectures import KPFCNN


def main():
    # Configure a light setup for a quick smoke test
    config = Toronto3DConfig()
    config.saving = False
    config.input_threads = 2
    config.in_radius = 2.0
    config.first_subsampling_dl = 0.12
    config.batch_num = 2
    config.epoch_steps = 2
    config.validation_size = 1
    config.max_epoch = 1

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate datasets
    print("Loading Toronto3D dataset (this may build caches on first run)...")
    t0 = time.time()
    train_ds = Toronto3DDataset(config, set='training', use_potentials=True)
    val_ds = Toronto3DDataset(config, set='validation', use_potentials=True)
    print(f"Datasets ready in {time.time()-t0:.1f}s | train clouds: {len(train_ds)}, val clouds: {len(val_ds)}")

    # Samplers and loaders
    train_sampler = Toronto3DSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=Toronto3DCollate,
        num_workers=config.input_threads,
        pin_memory=True,
    )

    # Fetch one batch
    print("Fetching one batch...")
    batch = next(iter(train_loader))
    if device.type == 'cuda':
        batch.to(device)

    # Build network
    net = KPFCNN(config, train_ds.label_values, train_ds.ignored_labels).to(device)
    net.train()

    # Forward and loss
    with torch.set_grad_enabled(True):
        outputs = net(batch, config)
        loss = net.loss(outputs, batch.labels)
        acc = net.accuracy(outputs, batch.labels)
        loss.backward()

    print(f"Forward OK: outputs={tuple(outputs.shape)}, loss={loss.item():.4f}, acc={acc*100:.1f}%")
    print("Smoke test completed.")


if __name__ == '__main__':
    try:
        main()
    except ModuleNotFoundError as e:
        print("Missing native extensions. Please build C++ wrappers first:")
        print("  1) cd cpp_wrappers\\cpp_subsampling; py setup.py build_ext --inplace")
        print("  2) cd ..\\cpp_neighbors; py setup.py build_ext --inplace")
        raise

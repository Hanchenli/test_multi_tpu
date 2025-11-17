# count_tpu_chips.py

import torch_xla
import torch_xla.core.xla_model as xm

def main():
    # Total number of TPU chips across all workers
    world_size = xm.xrt_world_size()

    # Chips available on this worker (local)
    local_world_size = xm.get_local_world_size()

    # Local device list (xla:0, xla:1, ...)
    devices = xm.get_xla_supported_devices()

    print("=== TPU Chip Information ===")
    print(f"Global world size (total chips): {world_size}")
    print(f"Local world size  (chips on this worker): {local_world_size}")
    print(f"Local devices: {devices}")
    print(f"My global ordinal: {xm.get_ordinal()}")
    print(f"My local ordinal:  {xm.get_local_ordinal()}")

if __name__ == "__main__":
    main()

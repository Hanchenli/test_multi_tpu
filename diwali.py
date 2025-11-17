from torch_xla import launch as xla_launch
from torch_xla.runtime import world_size as xla_world_size

def xla_main(*args):
    print("Happy Diwali!")
    print("Global world size:", xla_world_size())

if __name__ == "__main__":
    xla_launch(xla_main, nprocs=1)

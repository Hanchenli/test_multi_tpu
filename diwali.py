from torch_xla import launch as xla_launch
from torch_xla.runtime import world_size as xla_world_size


def main():
    print("Hanchen was here!")
    print(xla_world_size())


def xla_main(*args):
    main()


xla_launch(xla_main)

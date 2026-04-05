import sys
from omegaconf import OmegaConf, DictConfig


def load_config() -> DictConfig:
    config_path = None
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_path = arg.split("=")[1]
            break

    if config_path is None:
        raise ValueError("Missing --config=path/to/config.yaml")

    base = OmegaConf.load(config_path)
    cli = OmegaConf.from_cli()
    merged = OmegaConf.merge(base, cli)
    return merged

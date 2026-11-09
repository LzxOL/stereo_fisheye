"""正式训练的默认值集中放在 YAML；命令行参数可显式覆盖。"""

from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config") / "stereo_omnithings.yaml"


def load_config(path=None):
    with open(DEFAULT_CONFIG if path is None else path) as stream:
        return yaml.safe_load(stream)

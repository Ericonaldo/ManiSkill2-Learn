import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .base import BaseFileHandler


class YamlHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault("Loader", Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)

"""Stand-ins for the Decision Index kit's engine base, so that the engines can answer requests (load_sieve, decide)
without the kit installed. With the kit installed, the engines use the kit's own classes."""
import json


class Unsupported(ValueError):
    pass


def text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


class Engine:
    name = "engine"
    provenance = {}
    latency = "In-process request wall time including prompt construction; excludes model loading."

    def __init__(self, **options):
        self.options = options

    def runtime(self):
        return {}

    def synchronize(self):
        pass

    def close(self):
        pass

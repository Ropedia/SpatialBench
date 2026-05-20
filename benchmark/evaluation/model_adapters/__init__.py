ADAPTER_REGISTRY = {}


def register_adapter(name):
    """Decorator to register a model adapter."""
    def decorator(cls):
        ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter(name):
    """Get a registered model adapter by name."""
    if name not in ADAPTER_REGISTRY:
        available = list(ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
    return ADAPTER_REGISTRY[name]()

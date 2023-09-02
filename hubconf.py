dependencies = ['torch', 'scipy']



def tracker(config=None, checkpoint_path=None, map_location=None):
    """A full-blown XMem tracker.
    """
    from xmem import XMem
    # Call the model, load pretrained weights
    model = XMem(config or {}, checkpoint_path, map_location=map_location).eval()
    return model


def model(config={}, checkpoint_path=None, map_location=None):
    """The underlying XMem model. Use this if you want to implement your own tracker."""
    from xmem.model.network import XMem as XMemModel
    from xmem.checkpoint import ensure_checkpoint
    # Call the model, load pretrained weights
    model = XMemModel(config or {}).eval()
    checkpoint_path = checkpoint_path or ensure_checkpoint()
    model = XMemModel(config, checkpoint_path, map_location)
    model.eval()
    return model
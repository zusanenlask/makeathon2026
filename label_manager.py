class LabelManager:
    def __init__(self, prefix="This is a photo of "):
        self.prefix = prefix

    def add_CLIP_prefix(self, names):
        return [f"{self.prefix}{name}" for name in names]

    def strip_CLIP_prefix(self, label):
        if label.startswith(self.prefix):
            return label[len(self.prefix):]
        return label

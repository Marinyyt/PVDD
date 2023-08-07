class DistortionBase:

    def __init__(self, config) -> None:
        self.config = config
        # 用于记录退化力度分数的字典
        self.distortion_record = {}

    def __call__(self, img):
        raise NotImplementedError("degrade is not implemented.")

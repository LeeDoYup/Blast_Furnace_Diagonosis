import yaml


class VGGConf:
    conf = None

    @staticmethod
    def get(cfg=None):
        if cfg is not None:
            VGGConf.conf = VGGConf(cfg)
        return VGGConf.conf

    @staticmethod
    def g():
        return VGGConf.get()

    def __init__(self, path):
        if path:
            with open(path, 'r') as fp:
                self.conf = yaml.load(fp)

    def __getitem__(self, key):
        return self.conf[key]

if __name__ == '__main__':
    cf = VGGConf.get('conf/conf.yaml')
    pass

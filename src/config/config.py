import os
from mmengine import Config as MMConfig
from argparse import Namespace

from src.utils import assemble_project_path, Singleton

def process_general(config: MMConfig) -> MMConfig:

    config.workdir = assemble_project_path(config.workdir)
    os.makedirs(config.workdir, exist_ok=True)

    config.log_path = os.path.join(config.workdir, getattr(config, 'log_path', 'agent.log'))

    return config

class Config(MMConfig, metaclass=Singleton):
    def __init__(self):
        super(Config, self).__init__()

    def init_config(self, config_path: str, args: Namespace) -> None:
        # Initialize the general configuration
        mmconfig = MMConfig.fromfile(filename=assemble_project_path(config_path))
        if 'cfg_options' not in args or args.cfg_options is None:
            cfg_options = dict()
        else:
            cfg_options = args.cfg_options
        for item in args.__dict__:
            if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
                cfg_options[item] = args.__dict__[item]
        mmconfig.merge_from_dict(cfg_options)

        # Process general configuration
        mmconfig = process_general(mmconfig)

        self.__dict__.update(mmconfig.__dict__)

config = Config()
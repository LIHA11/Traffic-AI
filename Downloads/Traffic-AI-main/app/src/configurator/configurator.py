import json
import logging
from sys import platform
from os import path
import pathlib
import config_agent

from src.configurator.config import Config

class Configurator:
    _proj_root = path.split(pathlib.Path(__file__).parent.parent.parent.resolve())[0]

    @staticmethod
    def load_config():
        if Configurator.is_local():
            logging.info("Using local config file on Windows.")
            local_config_path = {
                "static": {
                    "json_config.json": path.join(
                        Configurator._proj_root, "config", "json_config_static.json"
                    ),
                },
                "services": path.join(
                    Configurator._proj_root, "config", "mock-service-instances.json"
                ),
            }
            config_agent.load_config(local_config_path)
        else:
            logging.info("Using config agent on Linux.")
            config_agent.load_config()

    @staticmethod
    def is_local():
        return platform == "win32" or platform == "linux"

    @staticmethod
    def get_config() -> Config:
        return json.loads(config_agent.get_config("json_config.json"))

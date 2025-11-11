#!env python
import asyncio
import logging

from src.logging.logger import Logger
from src.configurator.configurator import Configurator
from src.connector.connector import Connector
from src.quart.quart import start_quart_app

logger = logging.getLogger(__name__)

def exception_handler(_, context):
    logger.error(context)


async def main():
    
    Configurator.load_config()
    Logger.initialize(
        level=Configurator.get_config()["logging"]["level"]
    )
    await Connector.initiate()

    logger.info("Starting Sana LLM Traffic Copilot...")

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(exception_handler)

    await start_quart_app()


if __name__ == "__main__":
    asyncio.run(main())

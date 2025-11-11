import os
import logging
import ssl
import urllib3

logger = logging.getLogger(__name__)


def fix_local_ssl():

    os.environ["PYTHONHTTPSVERIFY"] = "0"

    ssl._create_default_https_context = ssl._create_unverified_context

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        import requests
        requests.packages.urllib3.disable_warnings()
        old_merge = requests.Session.merge_environment_settings

        def new_merge(self, url, proxies, stream, verify, cert):
            settings = old_merge(self, url, proxies, stream, verify, cert)
            settings['verify'] = False
            return settings

        requests.Session.merge_environment_settings = new_merge
    except (ImportError, AttributeError):
        pass

    try:
        import httpx
        old_init = httpx.Client.__init__

        def new_init(self, *args, **kwargs):
            kwargs['verify'] = False
            return old_init(self, *args, **kwargs)

        httpx.Client.__init__ = new_init

        old_async_init = httpx.AsyncClient.__init__

        def new_async_init(self, *args, **kwargs):
            kwargs['verify'] = False
            return old_async_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = new_async_init
    except (ImportError, AttributeError):
        pass

    logger.info("the configuration for bypassing SSL locally")

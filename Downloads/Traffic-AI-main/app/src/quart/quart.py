import logging
from typing import List
from quart import Quart, jsonify, request, g
from hypercorn.asyncio import serve
from hypercorn.config import Config
from datetime import datetime, timezone
from src.connector.connector import Connector
from src.error.not_found_error import NotFoundError
from src.error.internal_server_error import InternalServerError
from src.common.dataclass.access_token import AccessToken
from src.error.session_expired_error import SessionExpiredError
from src.error.unauthorized_error import UnauthorizedError
from src.configurator.sana_gateway_config import SanaGatewayConfig
from src.configurator.configurator import Configurator
from src.conversations import conversations_routes
from mongoengine import DoesNotExist
import jwt
from quart_cors import cors
from quart_schema import QuartSchema

logger = logging.getLogger(__name__)

########## Serving ##########
PORT = 8080
HOST = "0.0.0.0"

app = Quart(__name__)
app = cors(app, allow_origin="*") 
QuartSchema(app)

conversations_routes.init_conversations_routes(app)


@app.errorhandler(UnauthorizedError)
def handle_unauthorized_error(error: UnauthorizedError):
    logger.error(f"[{error.__class__.__name__}] {error}", exc_info=True)
    return jsonify({"error": error.message}), error.code


@app.errorhandler(SessionExpiredError)
def handle_session_expired_error(error: SessionExpiredError):
    logger.error(f"[{error.__class__.__name__}] {error}", exc_info=True)
    return jsonify({"error": error.message}), error.code


@app.errorhandler(InternalServerError)
def handle_internal_server_error(error: InternalServerError):
    logger.error(f"[{error.__class__.__name__}] {error}", exc_info=True)
    return jsonify({"error": error.message}), error.code


@app.errorhandler(NotFoundError)
def handle_not_found_error(error: NotFoundError):
    logger.error(f"[{error.__class__.__name__}] {error}", exc_info=True)
    return jsonify({"error": error.message}), error.code


@app.errorhandler(DoesNotExist)
def handle_does_not_exist_error(error: DoesNotExist):
    logger.error(f"[{error.__class__.__name__}] {error}", exc_info=True)
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(Exception)
def handle_generic_error(error: Exception):
    logger.error(f"[{error.__class__.__name__}] {error}", exc_info=True)
    return jsonify({"error": f"[{error.__class__.__name__}] {error}"}), 500


@app.before_request 

async def handle_frm_sso(): 
    if request.path == "/health/liveness": 

        return 

    g.access_token = AccessToken.from_dict( 

        jwt.decode("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJhdjg4c19TWXJ6QWxuSGNPYUEyRDhvM3A3eVZxeDVJSlZaaHNuRWxxLUk4In0.eyJleHAiOjE3NjI0MDA5ODksImlhdCI6MTc2MjQwMDY4OSwiYXV0aF90aW1lIjoxNzYyMzg5ODg3LCJqdGkiOiJjOTNjNDJmYy05NmNiLTRhMGMtOWU5My01NGExMTY4MTBhZGQiLCJpc3MiOiJodHRwczovL2lhbWZ3LmhvbWUtbnAub29jbC5jb20vYXV0aC9yZWFsbXMvb29jbC1kZXYiLCJzdWIiOiI5ZDAzYmU1NC05ZTliLTRjMDUtYTcxMS1lZjVkZjNhOTYxM2UiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzYW5hLWd3LWRldiIsInNpZCI6IjVhM2ViNThmLTMzNDUtNDk2OC1hODgyLThlZDk1Yzc0NDFmYiIsImFsbG93ZWQtb3JpZ2lucyI6WyJodHRwczovL2NvYXN0YWwuc2FuYS1ucC5vb2NsLmNvbSIsImh0dHBzOi8vc2ltdWxhdG9yLnNhbmEtbnAub29jbC5jb20iLCJodHRwczovL3NhbmEtbGxtLXRmYy1jb3BpbG90LXdlYi1kZXYuYS5ob21lLW5wLm9vY2wuY29tIiwiaHR0cHM6Ly8qLXcxMS5jb3JwLm9vY2wuY29tOioiLCJodHRwczovL2FuYWx5dGljcy5zYW5hLW5wLm9vY2wuY29tIiwiaHR0cHM6Ly9oa2d3c2R2MDA2Mjcub29jbC5jb206KiIsImh0dHBzOi8vdG1wLnNhbmEtbnAub29jbC5jb20iLCJodHRwczovL3NhbmEtYWxsb2NhdGlvbi13ZWItY29zZGV2LmEuaG9tZS1ucC5vb2NsLmNvbSIsImh0dHBzOi8vc2FuYS1yZWFsdGltZS1jbXMtZGV2LmEuaG9tZS1ucC5vb2NsLmNvbSIsImh0dHBzOi8vY2hlbndhMi13MTEuY29ycC5vb2NsLmNvbTo1ODAwMCIsImh0dHBzOi8vbm9lZGV2Lm9vY2wuY29tIiwiaHR0cHM6Ly9hbGxvY2F0aW9uLnNhbmEtbnAub29jbC5jb20iLCJodHRwczovL3NhbmEtY29hc3RhbC1kZXYuYS5ob21lLW5wLm9vY2wuY29tIiwiaHR0cDovL2hrZ3dzZHYwMDYyNy5vb2NsLmNvbToqIiwiaHR0cDovL2hrZ3dzZHYwMDYyNzoqIiwiaHR0cHM6Ly9zYW5hLW5kci1kZXYuYS5ob21lLW5wLm9vY2wuY29tIiwiaHR0cHM6Ly9zYW5hLXN1cHBvcnQtd2ViLWNvc2Rldi5hLmhvbWUtbnAub29jbC5jb20iLCJodHRwczovL3NhbmEta29uZy1ndy1jb3NkZXYuYS5ob21lLW5wLm9vY2wuY29tIiwiaHR0cHM6Ly90b3Auc2FuYS1ucC5vb2NsLmNvbSIsImh0dHBzOi8vc2FuYS1hbGxvY2F0aW9uLXRlbXBsYXRlLWRldi5hLmhvbWUtbnAub29jbC5jb20iLCJodHRwczovL3NhbmEtZ3ctZGV2LmEuaG9tZS1ucC5vb2NsLmNvbSJdLCJzY29wZSI6Im9wZW5pZCBzYW5hLnNpbXVsYXRvci5icmUgc2FuYS5jb2FzdGFsLmFsbG9jYXRpb24uZm9ybS5hcHByb3ZlciBzYW5hLmFsbG9jYXRpb24udGVtcGxhdGUuZWRpdCBzYW5hLmRpZnlfbm9ybWFsX3VzZXIgc2FuYS51c2VyLnN1cGVyIHNhbmEuYWxsb2NhdGlvbi5zdXBwb3J0IHNhbmEuY29hc3RhbC51dGlsaXphdGlvbi52aWV3IHNhbmEuY29hc3RhbC5hbGxvY2F0aW9uLmVkaXQgc2FuYS50bXAuc3VwcG9ydCBzYW5hLnRyYW5zaXRvcHRpbWEuc3VwcG9ydCBzYW5hLmdlbmVyYWwudGNyIHNhbmEudG1wIHNhbmEuY29hc3RhbC5wcmVkZWZpbmUudmlldyBwcm9maWxlIHNhbmEudHJhbnNpdG9wdGltYS5icmUgc2FuYS5zaW11bGF0b3Iuc3VwcG9ydCBzYW5hLmNvYXN0YWwucm91dGUuY2hlY2tib3ggc2FuYS5jb2FzdGFsLnN1cHBvcnQgc2FuYS5zcGFjZV9hcHByb3ZhbCBzYW5hLmNvYXN0YWwuZnVsZmlsbG1lbnQudmlldyBzYW5hLmFsbG9jYXRpb24udGVtcGxhdGUudmlldyBlbWFpbCBzYW5hLmNvYXN0YWwuYWxsb2NhdGlvbi5mb3JtLnJlcXVlc3RvciBzYW5hLmNvYXN0YWwuYWxsb2NhdGlvbi52aWV3IHNhbmEudHJhbnNpdG9wdGltYS52aWV3IHNhbmEuZGlmeV9hZG1pbiBzYW5hLmJhY3AucG9ydF9tb3ZlLnZpZXcgc2FuYS5nZW5lcmFsLnN0Y2Mgc2FuYS5jb2FzdGFsLnJvdXRlLnZpZXcgc2FuYS5kaWZ5X2VkaXRvciIsInRlbmFudF9pZCI6ImM0MDhiMWY4LTNlMDQtNDZhMi1hOWM3LWU2M2JiMmVjM2UyMiIsImF1ZCI6WyJzYW5hLWd3LWRldiIsIm5vZXZ0X2Rldl9jbGllbnQiLCJJQU1fUE9SVEFMIl0sInVwbiI6IkxJSEExMUBPT0NMLkNPTSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6IkhBUlJZIExJIiwicHJlZmVycmVkX3VzZXJuYW1lIjoibGloYTExQG9vY2wuY29tIiwiZ2l2ZW5fbmFtZSI6IkhBUlJZIiwiZmFtaWx5X25hbWUiOiJMSSIsImVtYWlsIjoiaGFycnkubGlAb29jbC5jb20ifQ.SZcicCPJ0pts7SG9thpe7dd-8M4ecdarvJ2r21o0yg_ErTMlVDU0LfHL5on3KHPovRo2OhmcY02vhbx6hTQ1Ulh1w16ZE7vwWIvL53TTBfskF6UeqbEblGVFlYp_Z3ScPvMJCt1yHp9J7RXF9Qfom3S-6RU4mPkq2DjpKAFMZhVLv6orE9XeowBa8hX9s48lYhEWe8AzL1Y3Ek17h11p2t32eknUAB6VJJMLE567FGVBFggbkI09FcYhPqgbx2PgcLFgzFtG81Mh8EKHGg8QZNe1pSU3tEYZIBuWawh4GaCFOS4fTlWpZUPwproevkpu809oQWbnYi5lm4nHXu7Mrw", 

            options={"verify_signature": False}, 

        ) 

    ) 

    g.user_id = g.access_token.get_user_id()


async def start_quart_app():
    logger.info("Starting Quart app...")

    config = Config()
    config.bind = [f"{HOST}:{PORT}"]

    await serve(app, config)


@app.get("/health/liveness")
def liveness():
    return jsonify({"status": "alive", "timestamp": datetime.now().isoformat()})

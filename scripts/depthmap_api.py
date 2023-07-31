# Non-public API. Don't host publicly - SECURITY RISKS!
# (will only be on with --api starting option)
# Currently no API stability guarantees are provided - API may break on any new commit.

from src import backbone
from api import api_routes

try:
    import modules.script_callbacks as script_callbacks
    if backbone.get_cmd_opt('api', False):
        script_callbacks.on_app_started(api_routes.depth_api)
except:
    print('DepthMap API could not start')

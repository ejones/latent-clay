import argparse
import concurrent.futures
import http.server
import importlib
import inspect
import json
import os
import os.path
import shlex
import sys
import traceback
from typing import Optional
import urllib.parse
import uuid

from PIL import Image
import yaml

from latentclay.capabilities import Capabilities
from latentclay.worldlog import WorldLog
from latentclay.commands import create_command_parser, handle_command_sync, run_command

_world_log = None


def resolve_config_obj(config: dict, device: Optional[str]):
    mod_name, cls_name = config['target'].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ValueError(f'{mod_name}.{cls_name} does not exist')

    params = {'device': device, **config['params']}
    try:
        inspect.signature(cls).bind(**params)
    except TypeError:
        params = config['params']

    return cls(**params)


def run_command_task(id, args, capabilities):
    try:
        run_command(id, args, capabilities, _world_log)
    except:
        traceback.print_exc()
        raise


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    executor = None
    cmd_parser = None
    capabilities = None

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server, directory='public')

    def do_GET(self):
        if self.path == '/world' or self.path.startswith('/world?'):
            purl = urllib.parse.urlparse(self.path)
            after_arg = urllib.parse.parse_qs(purl.query).get('after')
            after = int(after_arg[0]) if after_arg else 0
            resp = json.dumps(_world_log[after:]).encode('utf8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers['content-length'])
            body_str = self.rfile.read(length).decode('utf8')
            id = str(uuid.uuid4())
            if self.path == '/':
                try:
                    args = self.cmd_parser(body_str)
                except argparse.ArgumentError:
                    traceback.print_exc()
                    msg = b'Malformed or unsupported command'
                    self.send_response(400)
                    self.send_header('Content-Type', 'text/plain')
                    self.send_header('Content-Length', str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                handle_command_sync(id, args, _world_log)
                self.executor.submit(run_command_task, id, args, self.capabilities)
            else:
                self.send_response(404)
                self.end_headers()
                return
        except Exception:
            self.send_error(500)
            raise

        self.send_response(204)
        self.end_headers()


def main(args=sys.argv[1:]):
    global _world_log

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', help='config file location (YAML)')
    parser.add_argument('--device', help='Default GPU device (mps, cuda) for local inference or cpu')
    parser.add_argument('--port', type=int, default=8000)
    pargs = parser.parse_args(args)

    os.makedirs(os.path.join('public', 'output'), exist_ok=True)

    with open(pargs.config) as f:
        config = yaml.safe_load(f)

    _world_log = WorldLog(config['worldlog'])
    capabilities_dict = {
        k: resolve_config_obj(v, device=pargs.device) for k, v in config['capabilities'].items()
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        RequestHandler.executor = executor
        RequestHandler.cmd_parser = staticmethod(create_command_parser())
        RequestHandler.capabilities = Capabilities(**capabilities_dict)
        server = http.server.HTTPServer(('127.0.0.1', pargs.port), RequestHandler)
        print(f'serving on http://localhost:{pargs.port}')
        server.serve_forever()


if __name__ == "__main__":
    main()

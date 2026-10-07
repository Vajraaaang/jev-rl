import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent


class ServiceTests(unittest.TestCase):
    def test_http_fault_recovery_and_invalid_body(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen([sys.executable, '-m', 'jev_recovery.cli', 'serve', '--port', str(port)],
                                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f'http://127.0.0.1:{port}'
        def call(route, data=None):
            req = Request(base + route, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json'})
            with urlopen(req, timeout=2) as response:
                return json.load(response)
        try:
            for _ in range(100):
                try:
                    call('/health')
                    break
                except (URLError, ConnectionError):
                    time.sleep(.02)
            else:
                self.fail('HTTP service did not start')
            with self.assertRaises(HTTPError) as error:
                call('/reset', [])
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            call('/reset', {'faults': ['connections']})
            with self.assertRaises(HTTPError) as error:
                call('/request')
            self.assertEqual(error.exception.code, 503)
            error.exception.close()
            self.assertIn('pool', call('/logs')['logs'])
            call('/action', {'action': 'rollback'})
            call('/action', {'action': 'restart'})
            result = call('/action', {'action': 'stop'})
            self.assertTrue(result['success'])
            self.assertTrue(call('/health')['healthy'])
        finally:
            process.terminate()
            process.wait(timeout=5)

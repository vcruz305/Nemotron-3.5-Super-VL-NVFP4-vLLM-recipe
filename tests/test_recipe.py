import http.server
import json
import os
import re
import subprocess
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IS_BOLT = 'Bolt' in ROOT.name
MODEL = 'nemotron-bolt' if IS_BOLT else 'nemotron-super-vl'
CTX = 262144 if IS_BOLT else 1048576


class FakeEndpoint(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/v1/models':
            body = json.dumps({'data': [{'id': MODEL, 'max_model_len': CTX}]}).encode()
        elif self.path == '/metrics':
            body = b'vllm:spec_decode_num_draft_tokens_total{engine="0"} 3\n'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_error(404)
            return
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert payload['model'] == MODEL and payload['stream'] is True
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {"choices": []}\n\ndata: [DONE]\n\n')


class RecipeTests(unittest.TestCase):
    def test_shell_syntax_and_profiles(self):
        for script in ('serve.sh', 'bench.sh'):
            subprocess.run(['bash', '-n', str(ROOT / script)], check=True)
        profiles = ('dflash', 'no-draft') if IS_BOLT else ('mtp', 'dflash')
        for profile in profiles:
            p = subprocess.run(['bash', str(ROOT / 'serve.sh'), profile],
                               env={**os.environ, 'DRY_RUN': '1'},
                               text=True, capture_output=True, check=True)
            self.assertIn(f'context={CTX}', p.stdout)
            self.assertIn('MemoryMax=102005473280', p.stdout)
            if profile == 'no-draft':
                self.assertNotIn('speculative-config', p.stdout)
            else:
                self.assertIn('speculative-config', p.stdout)
            if not IS_BOLT:
                self.assertIn('gpu-memory-utilization', p.stdout)
                self.assertIn('0.74' if profile == 'dflash' else '0.71', p.stdout)
        p = subprocess.run(['bash', str(ROOT / 'bench.sh')],
                           env={**os.environ, 'DRY_RUN': '1'},
                           text=True, capture_output=True, check=True)
        for arg in ('--samples 32', '--curve-seconds 300', '--max-seconds 1200', '--policy strict'):
            self.assertIn(arg, p.stdout)

    def test_receipts_and_quality(self):
        quality = json.loads((ROOT / 'bench/quality-summary.json').read_text())
        self.assertEqual(sum(quality['n'].values()), 120)
        self.assertTrue(quality['complete'])
        self.assertEqual(quality['overall']['score'], 77.5 if IS_BOLT else 84.16666666666667)
        for name in ('baseline', 'dflash'):
            path = ROOT / 'bench/receipts' / (name + '.json')
            text = path.read_text()
            self.assertNotRegex(text, re.compile(r'/(home|root)/|nvapi-|10\.80\.|(?<![A-Za-z])SAGE(?![A-Za-z])'))
            data = json.loads(text)
            self.assertEqual(data['model'], MODEL)
            self.assertEqual(set(data['profiles']), {'decode', 'balanced', 'prefill'})
            self.assertEqual(data['server_props']['props']['data'][0]['root'], '${MODEL_DIR}')
            self.assertEqual(data['server_props']['openai_models']['data'][0]['root'], '${MODEL_DIR}')
            for profile, val in data['profiles'].items():
                self.assertEqual(val['confirmation']['requested'], 32)
                if name == 'dflash' or not IS_BOLT:
                    self.assertEqual(val['confirmation']['succeeded'], 32)

    def test_smoke_on_fake_endpoint(self):
        server = http.server.HTTPServer(('127.0.0.1', 0), FakeEndpoint)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            p = subprocess.run([sys.executable, str(ROOT / 'scripts/smoke.py'),
                                '--base-url', f'http://127.0.0.1:{server.server_port}/v1',
                                '--model', MODEL, '--context', str(CTX), '--expect-spec'],
                               text=True, capture_output=True, check=True, timeout=10)
            self.assertIn('OK model=' + MODEL, p.stdout)
        finally:
            server.shutdown()
            worker.join(timeout=2)
            server.server_close()


if __name__ == '__main__':
    unittest.main()

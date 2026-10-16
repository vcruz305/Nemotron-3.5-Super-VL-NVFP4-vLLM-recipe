#!/usr/bin/env python3
"""Check the exact local route and complete one streaming request."""
import argparse
import json
import urllib.request


def probe(base, model, context, expect_spec):
    base = base.rstrip('/')
    with urllib.request.urlopen(base + '/models', timeout=10) as r:
        data = json.load(r)['data']
    if len(data) != 1 or data[0]['id'] != model:
        raise RuntimeError('unexpected model IDs: ' + repr([x['id'] for x in data]))
    if data[0].get('max_model_len') != context:
        raise RuntimeError('unexpected advertised context: ' + repr(data[0].get('max_model_len')))
    payload = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': 'Say ready.'}],
        'max_tokens': 32,
        'temperature': 0,
        'stream': True,
    }).encode()
    req = urllib.request.Request(base + '/chat/completions', data=payload,
        headers={'Content-Type': 'application/json'}, method='POST')
    events = 0
    done = False
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if line.strip() == b'data: [DONE]':
                done = True
                break
            if line.startswith(b'data: '):
                json.loads(line[6:])
                events += 1
    if not (events > 0 and done):
        raise RuntimeError(f'incomplete SSE: events={events}, done={done}')
    if expect_spec:
        with urllib.request.urlopen(base.removesuffix('/v1') + '/metrics', timeout=10) as r:
            metrics = r.read().decode()
        matches = [l for l in metrics.splitlines()
                   if l.startswith('vllm:spec_decode_num_draft_tokens_total{')]
        if not matches or not any(float(l.rsplit(' ', 1)[-1]) > 0 for l in matches):
            raise RuntimeError('speculative draft counter did not increase')
    print(f'OK model={model} context={context} SSE_events={events} draft={expect_spec}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--model', required=True)
    p.add_argument('--context', type=int, required=True)
    p.add_argument('--expect-spec', action='store_true')
    args = p.parse_args()
    probe(args.base_url, args.model, args.context, args.expect_spec)

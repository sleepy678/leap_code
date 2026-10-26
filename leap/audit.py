from __future__ import annotations
import hashlib
import json
import math
import os
import re
import urllib.request
import numpy as np
import torch
from leap.data import TOPIC_NAMES
from leap.records import CATEGORY_NAMES, NEU

def audit_value(cand_edges: torch.Tensor, is_observed: torch.Tensor, n: int, eta_plus: float, eps: float=1e-08) -> torch.Tensor:
    dev = cand_edges.device
    deg_c = torch.zeros(n, device=dev).index_add_(0, torch.cat([cand_edges[:, 0], cand_edges[:, 1]]), torch.ones(2 * cand_edges.shape[0], device=dev))
    d_t = 1.0 + deg_c
    raw = (d_t[cand_edges[:, 0]] * d_t[cand_edges[:, 1]]).pow(-0.5)
    lo, hi = (raw.min(), raw.max())
    nu = (raw - lo) / (hi - lo + eps)
    return nu + eta_plus * (~is_observed).float()

def query_priority(c_in: torch.Tensor, u_in: torch.Tensor, nu: torch.Tensor, lambda_q: float, lambda_v: float) -> torch.Tensor:
    return u_in + lambda_q * (1.0 - (2.0 * c_in - 1.0).abs()) + lambda_v * nu

def select_queries(pi: torch.Tensor, r_audit: float) -> torch.Tensor:
    m = pi.shape[0]
    b = min(m, math.ceil(r_audit * m))
    if b <= 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.topk(pi, b).indices
TITLE_LIMIT = 300
ABSTRACT_LIMIT = 1400
PROMPT_TEMPLATE = 'You audit one candidate edge of a citation-style graph.\nCandidate ({source}): a relation between the two texts below.\n\nNode A: {text_a}\n\nNode B: {text_b}\n\nJudge from the texts alone: is this relation supportive, neutral, or risky for passing messages between the two nodes?\nAnswer with a single JSON object:\n{{"verdict": "supportive|neutral|risky", "scores": {{"supportive": s1, "neutral": s2, "risky": s3}}, "rationale": "one sentence"}}\nwith non-negative scores summing to 1.'

def format_endpoint(text: str) -> str:
    title, _, abstract = text.partition('\n')
    out = title[:TITLE_LIMIT]
    if abstract:
        out += ' | ' + abstract[:ABSTRACT_LIMIT]
    return out

def build_prompt(text_i: str, text_j: str, source: str) -> str:
    return PROMPT_TEMPLATE.format(source=source, text_a=format_endpoint(text_i), text_b=format_endpoint(text_j))

def parse_auditor_response(raw: str) -> dict:
    neutral = np.zeros(3)
    neutral[NEU] = 1.0
    fail = {'tau_out': neutral, 'ok': False, 'rationale': ''}
    if not isinstance(raw, str) or not raw.strip():
        return {**fail, 'parser_status': 'empty'}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search('\\{.*\\}', raw, re.S)
        if match is None:
            return {**fail, 'parser_status': 'json_error'}
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {**fail, 'parser_status': 'json_error'}
    scores = obj.get('scores', obj)
    try:
        vals = np.array([float(scores[name]) for name in CATEGORY_NAMES])
    except (KeyError, TypeError, ValueError):
        return {**fail, 'parser_status': 'score_error'}
    vals = np.clip(vals, 0.0, None)
    if not np.isfinite(vals).all() or vals.sum() <= 0:
        return {**fail, 'parser_status': 'score_error'}
    return {'tau_out': vals / vals.sum(), 'ok': True, 'parser_status': 'ok', 'rationale': str(obj.get('rationale', ''))}

class BaseAuditor:
    name = 'base'

    def audit_pair(self, text_i: str, text_j: str, source: str) -> str:
        raise NotImplementedError

class MockTopicAuditor(BaseAuditor):
    name = 'mock-topic'

    def __init__(self, topics=TOPIC_NAMES):
        self.topics = list(topics)
        self.call_count = 0

    def _topic(self, text: str):
        low = text.lower()
        for idx, phrase in enumerate(self.topics):
            if phrase in low:
                return idx
        return None

    def audit_pair(self, text_i: str, text_j: str, source: str) -> str:
        self.call_count += 1
        ti, tj = (self._topic(text_i), self._topic(text_j))
        if ti is None or tj is None:
            scores, verdict = ((0.2, 0.6, 0.2), 'neutral')
        elif ti == tj:
            scores, verdict = ((0.9, 0.07, 0.03), 'supportive')
        else:
            scores, verdict = ((0.05, 0.15, 0.8), 'risky')
        return json.dumps({'verdict': verdict, 'scores': dict(zip(CATEGORY_NAMES, scores)), 'rationale': f'topics: {ti} vs {tj} ({source})'})

class CosineAuditor(BaseAuditor):
    name = 'bow-cosine'

    def __init__(self, temperature: float=0.15):
        self.temperature = temperature
        self.call_count = 0
        self._bow_memo: dict = {}

    def _bow(self, text: str) -> dict:
        cached = self._bow_memo.get(text)
        if cached is None:
            counts: dict = {}
            for token in re.findall('[a-z]+', text.lower()):
                counts[token] = counts.get(token, 0) + 1
            cached = self._bow_memo[text] = counts
        return cached

    def audit_pair(self, text_i: str, text_j: str, source: str) -> str:
        self.call_count += 1
        a, b = (self._bow(text_i), self._bow(text_j))
        num = sum((v * b.get(k, 0) for k, v in a.items()))
        den = math.sqrt(sum((v * v for v in a.values()))) * math.sqrt(sum((v * v for v in b.values())))
        sim = num / den if den > 0 else 0.0
        logits = np.array([sim, 0.5, 1.0 - sim]) / self.temperature
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        return json.dumps({'verdict': CATEGORY_NAMES[int(scores.argmax())], 'scores': dict(zip(CATEGORY_NAMES, scores.round(4).tolist())), 'rationale': f'bag-of-words cosine {sim:.3f} ({source})'})

class HTTPAuditor(BaseAuditor):

    def __init__(self, base_url: str | None=None, api_key: str | None=None, model: str | None=None, timeout: float=60.0):
        self.base_url = (base_url or os.environ.get('LEAP_LLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4')).rstrip('/')
        self.api_key = api_key or os.environ.get('LEAP_LLM_API_KEY', '')
        self.model = model or os.environ.get('LEAP_LLM_MODEL', 'glm-5.1')
        self.timeout = timeout
        self.name = self.model
        self.call_count = 0

    def audit_pair(self, text_i: str, text_j: str, source: str) -> str:
        self.call_count += 1
        payload = json.dumps({'model': self.model, 'messages': [{'role': 'user', 'content': build_prompt(text_i, text_j, source)}], 'temperature': 0.0}).encode()
        req = urllib.request.Request(f'{self.base_url}/chat/completions', data=payload, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body['choices'][0]['message']['content']

class AuditCache:

    def __init__(self, path: str | None=None):
        self.path = path
        self.store: dict = {}
        self._dirty = False
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                self.store = json.load(fh)

    @staticmethod
    def key(text_i: str, text_j: str, source: str, auditor_name: str) -> str:
        blob = '||'.join([text_i, text_j, source, auditor_name])
        return hashlib.sha1(blob.encode()).hexdigest()

    def get(self, key: str):
        return self.store.get(key)

    def put(self, key: str, record: dict):
        self.store[key] = record
        self._dirty = True

    def flush(self):
        if self.path and self._dirty:
            tmp = f'{self.path}.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.store, fh, ensure_ascii=False, separators=(',', ':'))
            os.replace(tmp, self.path)
            self._dirty = False

def run_external_audit(records, texts, query_idx: torch.Tensor, auditor: BaseAuditor, cache: AuditCache | None=None) -> dict:
    from leap.internal import triplet_stats
    cache = cache or AuditCache()
    n_hit = n_fail = 0
    try:
        for idx in query_idx.tolist():
            i, j = (int(records.edges[idx, 0]), int(records.edges[idx, 1]))
            source = 'observed' if bool(records.is_observed[idx]) else 'retrieved'
            key = cache.key(texts[i], texts[j], source, auditor.name)
            entry = cache.get(key)
            if entry is None:
                try:
                    raw = auditor.audit_pair(texts[i], texts[j], source)
                except Exception as exc:
                    raw, status = ('', f'request_error:{type(exc).__name__}')
                    parsed = parse_auditor_response(raw)
                    parsed['parser_status'] = status
                else:
                    parsed = parse_auditor_response(raw)
                entry = {'edge': [i, j], 'source': source, 'tau_out': [float(v) for v in parsed['tau_out']], 'ok': bool(parsed['ok']), 'parser_status': parsed['parser_status'], 'rationale': parsed.get('rationale', ''), 'raw': raw}
                cache.put(key, entry)
            else:
                n_hit += 1
            tau = torch.tensor(entry['tau_out'], dtype=torch.float32)
            c, u = triplet_stats(tau)
            records.tau_out[idx] = tau.to(records.tau_out.device)
            records.c_out[idx] = c
            records.u_out[idx] = u
            records.m[idx] = bool(entry['ok'])
            n_fail += int(not entry['ok'])
    finally:
        cache.flush()
    return {'queried': int(query_idx.numel()), 'cache_hits': n_hit, 'failed': n_fail}

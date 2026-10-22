import json
import unittest
from fastapi.testclient import TestClient
from app import app, normalize, retrieve, postcheck

class DemoTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def events(self, question):
        r = self.client.post('/api/chat', json={'question': question})
        self.assertEqual(r.status_code, 200)
        return [(p.splitlines()[0][7:], json.loads(p.splitlines()[1][6:])) for p in r.text.strip().split('\n\n')]

    def test_alias_retrieval(self):
        q, changes = normalize('扑热息痛的注意事项')
        self.assertTrue(changes)
        self.assertEqual(retrieve(q)[0]['drug'], '对乙酰氨基酚')

    def test_interaction(self):
        self.assertIn('华法林', retrieve('布洛芬与华法林相互作用')[0]['drug'])

    def test_block_before_tokens(self):
        es = self.events('帮我诊断并开处方')
        self.assertFalse(any(k == 'sources' for k, _ in es))
        self.assertEqual(es[-1][1]['status'], 'blocked')

    def test_no_evidence(self):
        self.assertEqual(self.events('阿莫西林注意事项')[-1][1]['status'], 'insufficient')

    def test_postcheck_before_stream(self):
        es = self.events('扑热息痛的重复成分')
        checked = next(i for i,(k,d) in enumerate(es) if k=='stage' and d['name']=='后置审查')
        first_token = next(i for i,(k,d) in enumerate(es) if k=='token')
        self.assertLess(checked, first_token)
        self.assertEqual(es[-1][1]['status'], 'ok')

    def test_invalid_citation_and_dose(self):
        docs = retrieve('布洛芬')
        self.assertFalse(postcheck('服用 5mg [1]', docs))
        self.assertFalse(postcheck('资料 [99]', docs))

    def test_validation(self):
        self.assertEqual(self.client.post('/api/chat', json={'question':''}).status_code, 422)
        self.assertEqual(self.client.get('/').status_code, 200)

if __name__ == '__main__':
    unittest.main()

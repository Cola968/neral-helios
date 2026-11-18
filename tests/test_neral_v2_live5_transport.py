"""Regression: real V2 localhost client bootstraps browser session and accepts 201.

No credentials, model calls, Docker runs or network connections here.
"""
import json
import unittest
from unittest.mock import patch
from scripts.neral_v2_live5 import Live, _check_stdout


class Response:
    def __init__(self, status, payload=None):
        self.status = status
        self.body = json.dumps(payload or {}).encode("utf-8")
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def read(self, limit=0):
        return self.body[:limit] if limit else self.body


class FixtureOpener:
    def __init__(self):
        self.calls = []
    def open(self, request, timeout=0):
        self.calls.append((request.full_url, request.get_method(), timeout))
        path = request.full_url.split(":9690")[-1]
        if path == "/":
            return Response(200)
        if path == "/api/neral/preview/session":
            return Response(200, {"csrf": "x" * 32})
        if path == "/api/neral/account/register":
            return Response(201, {"authenticated": True, "csrf": "y" * 32})
        raise AssertionError("unexpected URL")


class ClientTransport(unittest.TestCase):
    def test_bootstrap_home_before_session_api(self):
        live=Live()
        op=FixtureOpener()
        live.opener=op
        live.bootstrap()
        result=live.call("/api/neral/preview/session")
        self.assertEqual(result["csrf"],"x"*32)
        self.assertEqual([x[0].split(":9690")[-1] for x in op.calls],
                         ["/", "/api/neral/preview/session"])

    def test_independently_check_exact_ordered_unlabelled_chip_triplet(self):
        gold={"Anna":9, "Ben":12, "Cora":24}
        _check_stdout("chips","9 12 24",gold)
        _check_stdout("chips","9, 12, 24",gold)
        _check_stdout("chips","Anna = 9; Ben = 12; Cora = 24",gold)
        with self.assertRaises(ValueError):
            _check_stdout("chips","9 12 25",gold)
        with self.assertRaises(ValueError):
            _check_stdout("chips","9 12 24 2",gold)
        with self.assertRaises(ValueError):
            _check_stdout("chips","12 9 24",gold)

    def test_post_registration_accepts_successful_201_created(self):
        live=Live()
        live.opener=FixtureOpener()
        live.csrf="x"*32
        result=live.call("/api/neral/account/register",
                         {"username":"testcase","password":"A"*20})
        self.assertTrue(result["authenticated"])


if __name__ == "__main__":
    unittest.main()

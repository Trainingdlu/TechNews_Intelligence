"""Unit tests for API security helpers and rate limiter cache controls."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

try:
    from tests.utils.bootstrap import ensure_agents_on_path
except ModuleNotFoundError:
    from utils.bootstrap import ensure_agents_on_path

ensure_agents_on_path()

# Removed heavy stub for agent so submodules are loadable.

# pydantic EmailStr requires email_validator and package version metadata.
email_validator_stub = types.ModuleType("email_validator")


class EmailNotValidError(ValueError):
    pass


def _fake_validate_email(email: str, *_args, **_kwargs):
    return types.SimpleNamespace(email=email, normalized=email)


email_validator_stub.EmailNotValidError = EmailNotValidError
email_validator_stub.validate_email = _fake_validate_email
sys.modules["email_validator"] = email_validator_stub

import pydantic.networks as pydantic_networks  # noqa: E402  pylint: disable=wrong-import-position

pydantic_networks.version = lambda _name: "2.0.0"

from app import api as api_mod  # noqa: E402  pylint: disable=wrong-import-position


class ApiSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_secret = api_mod.APPROVE_LINK_SECRET
        self._orig_ttl = api_mod.APPROVE_LINK_TTL_SEC
        self._orig_rate_window = api_mod.RATE_WINDOW
        self._orig_rate_limit = api_mod.RATE_LIMIT
        self._orig_request_log = api_mod._request_log  # pylint: disable=protected-access

    def tearDown(self) -> None:
        api_mod.APPROVE_LINK_SECRET = self._orig_secret
        api_mod.APPROVE_LINK_TTL_SEC = self._orig_ttl
        api_mod.RATE_WINDOW = self._orig_rate_window
        api_mod.RATE_LIMIT = self._orig_rate_limit
        api_mod._request_log = self._orig_request_log  # pylint: disable=protected-access

    def test_approve_signature_validation(self) -> None:
        api_mod.APPROVE_LINK_SECRET = "unit-test-secret"
        with patch.object(api_mod.time, "time", return_value=1000.0):
            sig = api_mod._build_approve_signature(7, 1060)  # pylint: disable=protected-access
            self.assertTrue(api_mod._is_valid_approve_signature(7, 1060, sig))  # pylint: disable=protected-access
            self.assertFalse(api_mod._is_valid_approve_signature(7, 999, sig))  # pylint: disable=protected-access
            self.assertFalse(api_mod._is_valid_approve_signature(7, 1060, sig + "00"))  # pylint: disable=protected-access

    def test_signed_approve_url_contains_exp_and_valid_signature(self) -> None:
        api_mod.APPROVE_LINK_SECRET = "another-secret"
        api_mod.APPROVE_LINK_TTL_SEC = 120

        with patch.object(api_mod.time, "time", return_value=2000.0):
            url = api_mod._build_signed_approve_url(12)  # pylint: disable=protected-access

        self.assertIsNotNone(url)
        self.assertIn("/approve/12", str(url))
        self.assertIn("exp=2120", str(url))

        query = str(url).split("?", maxsplit=1)[1]
        parts = dict(item.split("=", maxsplit=1) for item in query.split("&"))
        with patch.object(api_mod.time, "time", return_value=2000.0):
            self.assertTrue(api_mod._is_valid_approve_signature(12, int(parts["exp"]), parts["sig"]))  # pylint: disable=protected-access

    def test_confirmation_page_uses_post_form(self) -> None:
        html_doc = api_mod._render_approve_confirmation_page(9, 123456, "abc123")  # pylint: disable=protected-access
        self.assertIn('<form method="post"', html_doc)
        self.assertIn('/approve/9?exp=123456&sig=abc123', html_doc)

    def test_rate_limiter_cache_evicts_oldest_key(self) -> None:
        api_mod.RATE_LIMIT = 10
        api_mod.RATE_WINDOW = 60
        api_mod._request_log = api_mod._new_rate_log_cache(maxsize=2, ttl=60)  # pylint: disable=protected-access

        with patch.object(api_mod.time, "time", return_value=1000.0):
            api_mod._check_rate_limit("1.1.1.1")  # pylint: disable=protected-access
            api_mod._check_rate_limit("2.2.2.2")  # pylint: disable=protected-access
            api_mod._check_rate_limit("3.3.3.3")  # pylint: disable=protected-access

        self.assertEqual(len(api_mod._request_log), 2)  # pylint: disable=protected-access
        self.assertNotIn("1.1.1.1", api_mod._request_log)  # pylint: disable=protected-access

    def test_rate_limiter_filters_stale_timestamps(self) -> None:
        api_mod.RATE_LIMIT = 10
        api_mod.RATE_WINDOW = 10
        api_mod._request_log = api_mod._new_rate_log_cache(maxsize=10, ttl=60)  # pylint: disable=protected-access

        with patch.object(api_mod.time, "time", return_value=1000.0):
            api_mod._check_rate_limit("9.9.9.9")  # pylint: disable=protected-access

        with patch.object(api_mod.time, "time", return_value=1001.0):
            api_mod._check_rate_limit("9.9.9.9")  # pylint: disable=protected-access

        with patch.object(api_mod.time, "time", return_value=1012.0):
            api_mod._check_rate_limit("9.9.9.9")  # pylint: disable=protected-access

        self.assertEqual(api_mod._request_log["9.9.9.9"], [1012.0])  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()

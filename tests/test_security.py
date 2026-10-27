import unittest
from unittest.mock import patch

import httpx

from backend import main
from backend.security import (
    AccessTokenManager,
    InMemoryRateLimiter,
    configured_frontend_origins,
    normalize_origin,
    validate_access_password,
)


class SecurityConfigurationTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_frontend_origins(self) -> None:
        self.assertEqual(
            configured_frontend_origins(
                " https://RAG.example.com/ , https://rag.example.com "
            ),
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "https://rag.example.com",
            ],
        )

    def test_rejects_wildcard_credentials_paths_and_insecure_remote_origins(self) -> None:
        for origin in (
            "*",
            "https://user:password@example.com",
            "https://example.com/app",
            "http://example.com",
        ):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                normalize_origin(origin)

    def test_access_password_must_be_nonblank_and_long_enough(self) -> None:
        validate_access_password("")
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            validate_access_password(" " * 16)
        with self.assertRaisesRegex(ValueError, "at least 16 characters"):
            validate_access_password("short")
        validate_access_password("a-long-password-value")

    def test_access_tokens_are_short_lived_and_password_checked(self) -> None:
        sessions = AccessTokenManager(300)
        with self.assertRaisesRegex(ValueError, "Invalid access password"):
            sessions.issue("incorrect", "a-strong-backend-password")
        token, expires_at = sessions.issue(
            "a-strong-backend-password",
            "a-strong-backend-password",
        )

        self.assertGreater(expires_at, 0)
        self.assertTrue(sessions.validate(token))
        self.assertFalse(sessions.validate("unknown-session-token"))
        unicode_token, _ = sessions.issue(
            "pässword-is-long-enough",
            "pässword-is-long-enough",
        )
        self.assertTrue(sessions.validate(unicode_token))
        expired_sessions = AccessTokenManager(0)
        expired_token, _ = expired_sessions.issue("x", "x")
        self.assertFalse(expired_sessions.validate(expired_token))

    def test_rate_limiter_rejects_requests_over_the_configured_limit(self) -> None:
        limiter = InMemoryRateLimiter()
        self.assertEqual(
            limiter.allow("client", "query", limit=1, window_seconds=60),
            (True, 0),
        )
        allowed, retry_after = limiter.allow(
            "client",
            "query",
            limit=1,
            window_seconds=60,
        )
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)


class ApiSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_health_is_minimal_and_reports_authentication_requirement(self) -> None:
        response = await self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "authentication_required": main.AUTHENTICATION_ENABLED},
        )
        self.assertNotIn("filesystem", response.text.lower())
        self.assertNotIn("credential", response.text.lower())

    async def test_login_issues_session_and_api_requires_bearer_auth(self) -> None:
        sessions = AccessTokenManager(300)
        with (
            patch.object(main, "AUTHENTICATION_ENABLED", True),
            patch.object(main, "ACCESS_PASSWORD", "a-strong-backend-password"),
            patch.object(main, "ACCESS_TOKENS", sessions),
        ):
            denied = await self.client.get("/api/status")
            bad_login = await self.client.post(
                "/api/auth/token",
                json={"password": "wrong"},
            )
            login = await self.client.post(
                "/api/auth/token",
                json={"password": "a-strong-backend-password"},
            )
            token = login.json()["access_token"]
            allowed = await self.client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {token}"},
            )
            revoked = await self.client.post(
                "/api/auth/revoke",
                headers={"Authorization": f"Bearer {token}"},
            )
            denied_after_revoke = await self.client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {token}"},
            )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(bad_login.status_code, 401)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["token_type"], "bearer")
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(revoked.json(), {"revoked": True})
        self.assertEqual(denied_after_revoke.status_code, 401)

    async def test_indexing_and_atlassian_mutations_require_authentication(self) -> None:
        with (
            patch.object(main, "AUTHENTICATION_ENABLED", True),
            patch.object(main, "ACCESS_TOKENS", AccessTokenManager(300)),
        ):
            responses = [
                await self.client.post("/api/ingest", json={}),
                await self.client.post("/api/jira/sync", json={}),
                await self.client.post("/api/jira/disconnect", json={}),
                await self.client.post("/api/jira/connect", json={}),
            ]

        self.assertEqual([response.status_code for response in responses], [401] * 4)

    async def test_cors_preflight_allows_local_origin_but_not_unconfigured_origin(self) -> None:
        allowed = await self.client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        rejected = await self.client.get(
            "/api/health",
            headers={"Origin": "https://attacker.example"},
        )
        with (
            patch.object(main, "AUTHENTICATION_ENABLED", True),
            patch.object(main, "ACCESS_TOKENS", AccessTokenManager(300)),
        ):
            authentication_error = await self.client.get(
                "/api/status",
                headers={"Origin": "http://localhost:5173"},
            )

        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(
            allowed.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )
        self.assertNotIn("access-control-allow-origin", rejected.headers)
        self.assertEqual(authentication_error.status_code, 401)
        self.assertEqual(
            authentication_error.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    async def test_api_rejects_oversized_content_length(self) -> None:
        response = await self.client.post(
            "/api/auth/token",
            headers={"Content-Length": str(main.MAX_API_REQUEST_BYTES + 1)},
            content=b"",
        )

        self.assertEqual(response.status_code, 413)

    async def test_api_rejects_oversized_streamed_body_without_content_length(self) -> None:
        async def oversized_body():
            yield b"x" * 600_000
            yield b"x" * 600_000

        response = await self.client.post(
            "/api/auth/token",
            content=oversized_body(),
        )

        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()

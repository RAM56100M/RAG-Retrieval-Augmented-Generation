import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.atlassian import (
    AtlassianJiraConnector,
    JiraCredentials,
    confluence_page_text,
)


class AtlassianTests(unittest.TestCase):
    def test_confluence_storage_html_becomes_readable_text(self) -> None:
        text = confluence_page_text(
            {
                "body": {
                    "storage": {
                        "value": (
                            "<h2>Cart rules</h2>"
                            "<p>Use <strong>current</strong> inventory.</p>"
                            "<table><tr><th>Rule</th><td>Keep badge current.</td></tr></table>"
                        )
                    }
                }
            }
        )

        self.assertIn("Cart rules", text)
        self.assertIn("Use current inventory.", text)
        self.assertIn("Rule", text)
        self.assertIn("Keep badge current.", text)
        self.assertNotIn("<p>", text)
        self.assertEqual(confluence_page_text({"body": {}}), "")

    def test_next_page_url_rejects_cross_site_links(self) -> None:
        response = httpx.Response(
            200,
            headers={
                "Link": (
                    '<https://attacker.example/wiki/api/v2/pages?cursor=next>; '
                    'rel="next"'
                )
            },
            request=httpx.Request(
                "GET",
                "https://example.atlassian.net/wiki/api/v2/pages",
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "cross-site"):
            AtlassianJiraConnector._next_confluence_page_url(response, "pages")

    def test_next_page_url_accepts_same_site_link(self) -> None:
        response = httpx.Response(
            200,
            headers={
                "Link": (
                    '<https://example.atlassian.net/wiki/api/v2/pages?cursor=next>; '
                    'rel="next"'
                )
            },
            request=httpx.Request(
                "GET",
                "https://example.atlassian.net/wiki/api/v2/pages",
            ),
        )

        self.assertEqual(
            AtlassianJiraConnector._next_confluence_page_url(response, "pages"),
            "/wiki/api/v2/pages?cursor=next",
        )


class ConfluencePaginationTests(unittest.IsolatedAsyncioTestCase):
    async def test_confluence_permission_failure_does_not_mark_jira_disconnected(self) -> None:
        connector = AtlassianJiraConnector(
            lambda *_args, **_kwargs: 0,
            lambda *_args, **_kwargs: 0,
        )
        connector.status.update(stage="ready", connected=True)
        credentials = JiraCredentials(
            site_url="https://example.atlassian.net",
            email="user@example.com",
            api_token="test-token",
        )
        with patch.object(
            connector,
            "_list_confluence_spaces",
            new=AsyncMock(
                side_effect=ValueError(
                    "Confluence denied access. Check space permissions."
                )
            ),
        ):
            await connector._sync_confluence(credentials)

        self.assertTrue(connector.status["connected"])
        self.assertIsNone(connector.status["error"])
        self.assertEqual(
            connector.status["confluence_error"],
            "Confluence denied access. Check space permissions.",
        )

    async def test_pages_are_paginated_and_request_storage_bodies(self) -> None:
        requests: list[httpx.Request] = []
        base_url = "https://example.atlassian.net"

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.params.get("cursor") == "second":
                return httpx.Response(
                    200,
                    json={"results": [{"id": "2", "title": "Second page"}]},
                )
            return httpx.Response(
                200,
                headers={
                    "Link": (
                        f'<{base_url}/wiki/api/v2/pages?cursor=second>; rel="next"'
                    )
                },
                json={"results": [{"id": "1", "title": "First page"}]},
            )

        connector = AtlassianJiraConnector(lambda *_args, **_kwargs: 0, lambda *_args, **_kwargs: 0)
        async with httpx.AsyncClient(
            base_url=base_url,
            transport=httpx.MockTransport(handler),
        ) as client:
            pages = [
                page
                async for batch in connector._list_confluence_page_batches(client)
                for page in batch
            ]

        self.assertEqual([page["id"] for page in pages], ["1", "2"])
        self.assertEqual(requests[0].url.path, "/wiki/api/v2/pages")
        self.assertEqual(requests[0].url.params.get("body-format"), "storage")
        self.assertEqual(requests[0].url.params.get("status"), "current")
        self.assertEqual(requests[1].url.params.get("cursor"), "second")


if __name__ == "__main__":
    unittest.main()

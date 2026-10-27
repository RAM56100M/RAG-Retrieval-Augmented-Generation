import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import httpx
import keyring
from pydantic import BaseModel, Field, SecretStr

LOGGER = logging.getLogger(__name__)
KEYRING_SERVICE = "rag-explorer-atlassian-jira"
LEGACY_KEYRING_SERVICE = "rag-explorer-atlassian-mcp"
PAGE_SIZE = 100
ISSUE_FIELDS = [
    "summary",
    "description",
    "status",
    "issuetype",
    "priority",
    "project",
    "labels",
    "issuelinks",
    "created",
    "updated",
]


class JiraCredentials(BaseModel):
    site_url: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=320)
    api_token: SecretStr


class _ConfluenceStorageTextParser(HTMLParser):
    _BLOCK_TAGS = {
        "br",
        "dd",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "p",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, _attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def confluence_page_text(page: dict[str, Any]) -> str:
    body = page.get("body")
    if not isinstance(body, dict):
        return ""
    storage = body.get("storage")
    value = storage.get("value") if isinstance(storage, dict) else None
    if not isinstance(value, str):
        return ""
    parser = _ConfluenceStorageTextParser()
    parser.feed(value)
    return "\n".join(
        line.strip()
        for line in "".join(parser.parts).splitlines()
        if line.strip()
    )


def normalize_site_url(site_url: str) -> str:
    parsed = urlsplit(site_url.strip().rstrip("/"))
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or not hostname.endswith(".atlassian.net")
        or hostname == "atlassian.net"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Enter a Jira Cloud site URL such as https://your-company.atlassian.net."
        )
    return f"https://{hostname}"


class AtlassianJiraConnector:
    def __init__(
        self,
        index_issues: Callable[..., int],
        index_confluence_pages: Callable[..., int],
    ) -> None:
        self.index_issues = index_issues
        self.index_confluence_pages = index_confluence_pages
        self.status: dict[str, Any] = {
            "stage": "disconnected",
            "message": "Connect your Atlassian account to sync Jira and Confluence.",
            "connected": False,
            "project_count": 0,
            "issue_count": 0,
            "chunk_count": 0,
            "confluence_space_count": 0,
            "confluence_page_count": 0,
            "confluence_chunk_count": 0,
            "confluence_error": None,
            "error": None,
        }
        self._task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._credentials: JiraCredentials | None = None

    def snapshot(self) -> dict[str, Any]:
        return dict(self.status)

    async def start_on_startup(self) -> None:
        try:
            self._remove_legacy_oauth_credentials()
            stored = keyring.get_password(KEYRING_SERVICE, "jira-api")
            if not stored:
                return
            data = json.loads(stored)
            self._credentials = JiraCredentials.model_validate(data)
            normalize_site_url(self._credentials.site_url)
            self.status.update(
                stage="syncing",
                message="Using the saved Atlassian connection to refresh Jira and Confluence.",
                error=None,
            )
            self._start_sync(self._credentials)
        except Exception as exc:
            LOGGER.exception("Could not load saved Jira API credentials")
            self.status.update(
                stage="error",
                message="Could not load the saved Jira connection.",
                error=f"{type(exc).__name__}. Check backend logs and Credential Manager.",
            )

    async def connect(self, credentials: JiraCredentials) -> dict[str, Any]:
        async with self._connect_lock:
            if self._task is not None and not self._task.done():
                return self.snapshot()

            try:
                credentials = credentials.model_copy(
                    update={
                        "site_url": normalize_site_url(credentials.site_url),
                        "email": credentials.email.strip(),
                        "api_token": SecretStr(
                            credentials.api_token.get_secret_value().strip()
                        ),
                    }
                )
                if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", credentials.email):
                    raise ValueError("Enter the email address for your Atlassian account.")
                if not credentials.api_token.get_secret_value().strip():
                    raise ValueError("Enter an Atlassian API token.")
                self.status.update(
                    stage="connecting",
                    message="Checking your Atlassian API token.",
                    error=None,
                )
                await self._verify_credentials(credentials)
            except ValueError as exc:
                self.status.update(
                    stage="error",
                    message="Jira sign-in failed.",
                    error=str(exc),
                )
                raise

            try:
                self._save_credentials(credentials)
            except Exception:
                LOGGER.exception("Could not save Jira API credentials")
                self.status.update(
                    stage="error",
                    message="Could not save the Jira connection.",
                    error="Windows Credential Manager could not save the credentials.",
                )
                raise RuntimeError(
                    "Windows Credential Manager could not save the Jira credentials."
                ) from None

            self._credentials = credentials
            self.status.update(
                stage="syncing",
                message="Atlassian access verified. Loading Jira and Confluence content.",
                connected=False,
                error=None,
            )
            self._start_sync(credentials)
            return self.snapshot()

    async def disconnect(self) -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        for service, keys in (
            (KEYRING_SERVICE, ("jira-api",)),
            (LEGACY_KEYRING_SERVICE, ("tokens", "client")),
        ):
            for key in keys:
                if keyring.get_password(service, key) is not None:
                    keyring.delete_password(service, key)

        self._credentials = None
        self.status.update(
            stage="disconnected",
            message="Atlassian credentials removed. Previously indexed Jira issues and Confluence pages remain available locally.",
            connected=False,
            error=None,
        )
        return self.snapshot()

    async def sync(self) -> dict[str, Any]:
        if self._credentials is None:
            raise ValueError("Connect an Atlassian account before syncing.")
        if self._task is not None and not self._task.done():
            return self.snapshot()
        self.status.update(
            stage="syncing",
            message="Refreshing Jira issues and Confluence pages.",
            connected=False,
            error=None,
            confluence_error=None,
        )
        self._start_sync(self._credentials)
        return self.snapshot()

    async def shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _verify_credentials(self, credentials: JiraCredentials) -> None:
        try:
            async with self._client(credentials) as client:
                response = await client.get("/rest/api/3/myself")
                self._raise_for_jira_error(response)
                account = response.json()
                if not isinstance(account, dict) or not account.get("accountId"):
                    raise RuntimeError("Jira returned an unexpected account response.")
        except httpx.RequestError as exc:
            LOGGER.warning("Jira credential validation request failed: %s", type(exc).__name__)
            raise ValueError(
                "Could not reach that Jira site. Check the site URL and your connection."
            ) from None

    def _client(self, credentials: JiraCredentials) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=normalize_site_url(credentials.site_url),
            auth=(credentials.email, credentials.api_token.get_secret_value()),
            headers={"Accept": "application/json"},
            timeout=httpx.Timeout(60),
        )

    @staticmethod
    def _raise_for_jira_error(response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code == 401:
            raise ValueError(
                "Jira rejected the email or API token. Check both and try again."
            )
        if response.status_code == 403:
            raise ValueError(
                "Jira denied access. Check that the account has permission to browse projects and issues."
            )
        if response.status_code == 429:
            raise ValueError(
                "Jira rate limited the request. Wait a moment, then try again."
            )
        if response.status_code >= 500:
            raise ValueError(
                f"Jira is temporarily unavailable (HTTP {response.status_code}). Try again later."
            )
        raise ValueError(
            f"Jira rejected the request (HTTP {response.status_code}). Check the site URL and API token."
        )

    @staticmethod
    def _raise_for_confluence_error(response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code == 401:
            raise ValueError(
                "Confluence rejected the Atlassian API token. Check the email and token."
            )
        if response.status_code == 403:
            raise ValueError(
                "Confluence denied access. Check that the account can use Confluence and view the required spaces."
            )
        if response.status_code == 429:
            raise ValueError(
                "Confluence rate limited the request. Wait a moment, then try again."
            )
        if response.status_code >= 500:
            raise ValueError(
                f"Confluence is temporarily unavailable (HTTP {response.status_code}). Try again later."
            )
        raise ValueError(
            f"Confluence rejected the request (HTTP {response.status_code})."
        )

    def _save_credentials(self, credentials: JiraCredentials) -> None:
        keyring.set_password(
            KEYRING_SERVICE,
            "jira-api",
            json.dumps(
                {
                    "site_url": credentials.site_url,
                    "email": credentials.email,
                    "api_token": credentials.api_token.get_secret_value(),
                }
            ),
        )

    @staticmethod
    def _remove_legacy_oauth_credentials() -> None:
        for key in ("tokens", "client"):
            if keyring.get_password(LEGACY_KEYRING_SERVICE, key) is not None:
                keyring.delete_password(LEGACY_KEYRING_SERVICE, key)

    def _start_sync(self, credentials: JiraCredentials) -> None:
        self._task = asyncio.create_task(self._sync(credentials))

    async def _sync(self, credentials: JiraCredentials) -> None:
        try:
            async with self._client(credentials) as client:
                projects = await self._list_projects(client)
                self.status.update(
                    stage="syncing",
                    message=f"Found {len(projects)} accessible projects. Retrieving Jira issues.",
                    project_count=len(projects),
                    issue_count=0,
                    chunk_count=0,
                    error=None,
                )
                issue_count = 0
                chunk_count = 0
                seen_issue_keys: set[str] = set()
                page_number = 0
                async for page_issues in self._list_issue_pages(client, projects):
                    page_number += 1
                    for start in range(0, len(page_issues), 50):
                        issue_batch = page_issues[start : start + 50]
                        self.status.update(
                            stage="syncing",
                            message=f"Indexing Jira issues — {issue_count} indexed; processing batch {page_number}.",
                            project_count=len(projects),
                            issue_count=issue_count,
                            chunk_count=chunk_count,
                            error=None,
                        )
                        indexed_chunks = await asyncio.to_thread(
                            self.index_issues,
                            issue_batch,
                            projects,
                            normalize_site_url(credentials.site_url),
                            prune_stale=False,
                        )
                        issue_count += len(issue_batch)
                        chunk_count += indexed_chunks
                        seen_issue_keys.update(
                            key
                            for issue in issue_batch
                            if (key := issue.get("key"))
                        )
                        self.status.update(
                            stage="syncing",
                            message=f"Indexed {issue_count} Jira issues from {len(projects)} projects.",
                            project_count=len(projects),
                            issue_count=issue_count,
                            chunk_count=chunk_count,
                            error=None,
                        )

            self.status.update(
                stage="syncing",
                message="Finishing Jira index cleanup.",
                project_count=len(projects),
                issue_count=issue_count,
                chunk_count=chunk_count,
                error=None,
            )
            await asyncio.to_thread(
                self.index_issues,
                [],
                projects,
                normalize_site_url(credentials.site_url),
                retain_issue_keys=seen_issue_keys,
            )
            await self._sync_confluence(credentials)
            self.status.update(
                stage="ready",
                message=(
                    f"Indexed {issue_count} Jira issues and "
                    f"{self.status['confluence_page_count']} Confluence pages."
                ),
                connected=True,
                project_count=len(projects),
                issue_count=issue_count,
                chunk_count=chunk_count,
                error=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Jira synchronization failed")
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else f"{type(exc).__name__}. Check backend logs and Atlassian permissions."
            )
            self.status.update(
                stage="error",
                message="Atlassian connection or Jira synchronization failed.",
                connected=False,
                error=message,
            )

    async def _sync_confluence(
        self,
        credentials: JiraCredentials,
    ) -> None:
        page_count = 0
        chunk_count = 0
        try:
            async with self._client(credentials) as client:
                spaces = await self._list_confluence_spaces(client)
                spaces_by_id = {
                    str(space["id"]): space
                    for space in spaces
                    if space.get("id") is not None
                }
                seen_page_ids: set[str] = set()
                self.status.update(
                    stage="syncing",
                    message=f"Found {len(spaces)} Confluence spaces. Retrieving pages.",
                    confluence_space_count=len(spaces),
                    confluence_page_count=0,
                    confluence_chunk_count=0,
                    confluence_error=None,
                )
                async for page_batch in self._list_confluence_page_batches(client):
                    prepared_batch = []
                    for page in page_batch:
                        page_id = str(page.get("id", "")).strip()
                        if not page_id:
                            continue
                        page["space"] = spaces_by_id.get(str(page.get("spaceId")), {})
                        prepared_batch.append(page)
                        seen_page_ids.add(page_id)
                    if not prepared_batch:
                        continue
                    for start in range(0, len(prepared_batch), 25):
                        page_slice = prepared_batch[start : start + 25]
                        indexed_chunks = await asyncio.to_thread(
                            self.index_confluence_pages,
                            page_slice,
                            normalize_site_url(credentials.site_url),
                            prune_stale=False,
                        )
                        page_count += len(page_slice)
                        chunk_count += indexed_chunks
                        self.status.update(
                            stage="syncing",
                            message=f"Indexing Confluence pages — {page_count} indexed.",
                            confluence_space_count=len(spaces),
                            confluence_page_count=page_count,
                            confluence_chunk_count=chunk_count,
                            confluence_error=None,
                        )

                await asyncio.to_thread(
                    self.index_confluence_pages,
                    [],
                    normalize_site_url(credentials.site_url),
                    retain_page_ids=seen_page_ids,
                )
            self.status.update(
                confluence_space_count=len(spaces),
                confluence_page_count=page_count,
                confluence_chunk_count=chunk_count,
                confluence_error=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Confluence synchronization failed")
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else f"{type(exc).__name__}. Check backend logs and Confluence permissions."
            )
            self.status.update(
                confluence_page_count=page_count,
                confluence_chunk_count=chunk_count,
                confluence_error=message,
            )

    @staticmethod
    def _next_confluence_page_url(
        response: httpx.Response,
        resource: str,
    ) -> str | None:
        link = response.links.get("next")
        raw_url = link.get("url") if isinstance(link, dict) else None
        if not raw_url:
            return None
        parsed = urlsplit(str(raw_url))
        request_url = response.request.url
        if parsed.scheme and parsed.scheme.lower() != "https":
            raise RuntimeError("Confluence returned an insecure pagination URL.")
        if parsed.netloc and (
            parsed.hostname is None
            or parsed.hostname.lower() != request_url.host.lower()
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RuntimeError("Confluence returned a cross-site pagination URL.")
        if parsed.path.rstrip("/") != f"/wiki/api/v2/{resource}":
            raise RuntimeError("Confluence returned an unexpected pagination URL.")
        return f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path

    async def _list_confluence_spaces(
        self,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        spaces: list[dict[str, Any]] = []
        next_url: str | None = None
        seen_urls: set[str] = set()
        while True:
            response = await client.get(
                next_url or "/wiki/api/v2/spaces",
                params=None if next_url else {"limit": PAGE_SIZE, "status": "current"},
            )
            self._raise_for_confluence_error(response)
            page = response.json()
            values = page.get("results") if isinstance(page, dict) else None
            if not isinstance(values, list):
                raise RuntimeError("Confluence space search returned an unexpected response.")
            spaces.extend(space for space in values if isinstance(space, dict))
            next_url = self._next_confluence_page_url(response, "spaces")
            if not next_url:
                break
            if next_url in seen_urls:
                raise RuntimeError("Confluence space search returned a repeated page link.")
            seen_urls.add(next_url)
        return spaces

    async def _list_confluence_page_batches(
        self,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        next_url: str | None = None
        seen_urls: set[str] = set()
        while True:
            response = await client.get(
                next_url or "/wiki/api/v2/pages",
                params=(
                    None
                    if next_url
                    else {
                        "limit": PAGE_SIZE,
                        "status": "current",
                        "body-format": "storage",
                    }
                ),
            )
            self._raise_for_confluence_error(response)
            page = response.json()
            values = page.get("results") if isinstance(page, dict) else None
            if not isinstance(values, list):
                raise RuntimeError("Confluence page search returned an unexpected response.")
            pages = [item for item in values if isinstance(item, dict)]
            if pages:
                yield pages
            next_url = self._next_confluence_page_url(response, "pages")
            if not next_url:
                break
            if next_url in seen_urls:
                raise RuntimeError("Confluence page search returned a repeated page link.")
            seen_urls.add(next_url)

    async def _list_projects(
        self, client: httpx.AsyncClient
    ) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        start_at = 0
        while True:
            response = await client.get(
                "/rest/api/3/project/search",
                params={"startAt": start_at, "maxResults": PAGE_SIZE},
            )
            self._raise_for_jira_error(response)
            page = response.json()
            values = page.get("values") if isinstance(page, dict) else None
            if not isinstance(values, list):
                raise RuntimeError("Jira project search returned an unexpected response.")
            projects.extend(project for project in values if isinstance(project, dict))
            start_at += len(values)
            if page.get("isLast") is True or not values:
                break
            total = page.get("total")
            if isinstance(total, int) and start_at >= total:
                break
        return projects

    async def _list_issue_pages(
        self,
        client: httpx.AsyncClient,
        projects: list[dict[str, Any]],
    ) -> AsyncIterator[list[dict[str, Any]]]:
        project_keys = sorted(
            {
                str(project.get("key", "")).strip()
                for project in projects
                if project.get("key")
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(project["key"]).strip())
            }
        )
        if not project_keys:
            return

        quoted_keys = [f'"{key}"' for key in project_keys]
        jql = f"project in ({','.join(quoted_keys)}) ORDER BY updated DESC"
        next_page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            payload: dict[str, Any] = {
                "jql": jql,
                "fields": ISSUE_FIELDS,
                "maxResults": PAGE_SIZE,
            }
            if next_page_token:
                payload["nextPageToken"] = next_page_token
            response = await client.post("/rest/api/3/search/jql", json=payload)
            self._raise_for_jira_error(response)
            page = response.json()
            page_issues = page.get("issues") if isinstance(page, dict) else None
            if not isinstance(page_issues, list):
                raise RuntimeError("Jira issue search returned an unexpected response.")
            yield [issue for issue in page_issues if isinstance(issue, dict)]
            if page.get("isLast") is True:
                break
            token = page.get("nextPageToken")
            if not token:
                break
            token = str(token)
            if token in seen_tokens:
                raise RuntimeError("Jira issue search returned a repeated page token.")
            seen_tokens.add(token)
            next_page_token = token

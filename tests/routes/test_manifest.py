"""Tests for the web app manifest, its icons, and the HTML link."""

import json

from .conftest import _patched_client


def test_manifest_served_with_correct_media_type(tmp_path):
    """The manifest is served at the root with ``application/manifest+json``."""
    with _patched_client(tmp_path) as client:
        resp = client.get("/manifest.webmanifest")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/manifest+json")


def test_manifest_body_is_valid_and_references_icons(tmp_path):
    """The manifest parses as JSON and lists the three icon variants."""
    with _patched_client(tmp_path) as client:
        resp = client.get("/manifest.webmanifest")
    data = json.loads(resp.content)
    assert data["name"] == "Introspect"
    assert data["start_url"] == "/"
    srcs = {icon["src"] for icon in data["icons"]}
    assert srcs == {
        "/static/icons/icon-192.png",
        "/static/icons/icon-512.png",
        "/static/icons/icon-512-maskable.png",
    }
    assert any(icon.get("purpose") == "maskable" for icon in data["icons"])


def test_icons_served_as_png(tmp_path):
    """Each referenced icon is reachable and is a real PNG."""
    png_magic = b"\x89PNG\r\n\x1a\n"
    with _patched_client(tmp_path) as client:
        for name in ("icon-192.png", "icon-512.png", "icon-512-maskable.png"):
            resp = client.get(f"/static/icons/{name}")
            assert resp.status_code == 200, name
            assert resp.headers["content-type"] == "image/png", name
            assert resp.content.startswith(png_magic), name


def test_base_head_links_manifest(tmp_path):
    """Rendered pages advertise the manifest and a theme color."""
    with _patched_client(tmp_path) as client:
        html = client.get("/").text
    assert '<link rel="manifest" href="/manifest.webmanifest">' in html
    assert '<meta name="theme-color" content="#1a1a2e">' in html

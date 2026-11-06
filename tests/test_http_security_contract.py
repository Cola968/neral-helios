"""HTTP route inventory change gate; not proof of vulnerability absence."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# New routes require explicit review: auth, ownership, input and return data.
EXPECTED_API_ROUTES = {
    ("GET", "/api/projects"),
    ("POST", "/api/projects"),
    ("GET", "/api/projects/<project>"),
    ("POST", "/api/projects/<project>/<action>"),
    ("GET", "/api/info"),
    ("GET", "/api/metrics"),
    ("POST", "/api/model"),
    ("POST", "/api/cancel"),
    ("POST", "/api/generate"),
    ("GET", "/api/chats"),
    ("POST", "/api/chats/<key>"),
    ("GET", "/api/health"),
}


def http_routes(source: str) -> set[tuple[str, str]]:
    routes = set()
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            if not isinstance(decorator.func.value, ast.Name) or decorator.func.value.id != "app":
                continue
            method = decorator.func.attr.lower()
            if method not in {"get", "post", "delete", "put", "patch", "route"} or not decorator.args:
                continue
            path = ast.literal_eval(decorator.args[0])
            if method == "route":
                options = {key.arg: key.value for key in decorator.keywords}
                methods = ast.literal_eval(options.get("methods")) if "methods" in options else ["GET"]
            else:
                methods = [method.upper()]
            routes.update((m.upper(), path) for m in methods)
    return routes


def test_http_route_inventory_requires_review_on_changes():
    actual = http_routes((ROOT / "app.py").read_text(encoding="utf-8"))
    assert {(method, path) for method, path in actual if path.startswith("/api/")} == EXPECTED_API_ROUTES


def test_no_unreviewed_upload_or_billing_http_endpoints():
    actual = http_routes((ROOT / "app.py").read_text(encoding="utf-8"))
    assert not any("upload" in path or "webhook" in path or "checkout" in path for _, path in actual)


def test_local_api_uses_central_session_guard():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "local_and_rate_guard")
    assert any(isinstance(d, ast.Attribute) and d.attr == "before_request" for d in guard.decorator_list)
    fragment = ast.get_source_segment(source, guard)
    assert all(s in fragment for s in ("remote_addr", "request.host", "Origin", "SESSION_TOKEN", "compare_digest", "request.path.startswith('/api/')"))

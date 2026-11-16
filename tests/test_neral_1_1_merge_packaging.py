"""NERAL 1.1 release ZIP must include the optional runtime's direct dependencies."""
from scripts.build_neral_1_0_preview import REQUIRED_FILES


def test_adaptive_runtime_modules_are_packaged():
    needed = {
        "neral_adaptive_planner.py", "neral_adaptive_ollama.py",
        "neral_adaptive_bounded_fallback.py", "neral_python_assert_contract.py",
        "neral_python_source_router.py", "neral_python_post_source.py",
        "neral_python_math_verifier.py", "neral_python_logic_verifier.py",
        "neral_python_unique_logic.py",
    }
    assert needed <= set(REQUIRED_FILES)


def test_app_ui_and_security_modules_are_not_removed_from_package():
    assert {"neral_web_api.py", "neral_web_search.py",
            "neral_preview_server.py", "neral_local_accounts.py",
            "neral_tir_regression_guard.py"} <= set(REQUIRED_FILES)

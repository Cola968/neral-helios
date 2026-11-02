"""Opt-in launcher: python scripts/neral_control_helios.py --help.

No Helios module or model is imported until main() is explicitly called.
"""
from pathlib import Path
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run Helios with the opt-in Neral Control panel.')
    parser.add_argument('--allow-model-inference', action='store_true',
                        help='Allow explicit requests to already available local Qwen or enabled Ollama.')
    parser.add_argument('--project-root', type=Path, help='Explicit read-only project source root.')
    parser.add_argument('--enable-web-search', action='store_true',
                        help='Offer free Wikipedia search; each task must still opt in to the web tool.')
    parser.add_argument('--search-language', choices=('en','de'), default='de',
                        help='Wikipedia source language (default: de); no startup queries.')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if not args.allow_model_inference:
        # A tools-only launch must also disable legacy runtime auto-loading.
        os.environ["HELIOS_PREPARATION_MODE"] = "1"
    import app as helios
    from neral_control.helios import install
    search_provider = None
    if args.enable_web_search:
        from neral_control.search import WikipediaSearch
        search_provider = WikipediaSearch(enabled=True, language=args.search_language)
    install(helios, project_root=args.project_root, search_provider=search_provider,
            allow_model_inference=args.allow_model_inference)
    helios.app.run(host='127.0.0.1', port=helios.APP_PORT, threaded=True, debug=False)


if __name__ == '__main__':
    main()

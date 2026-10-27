"""Local/mobile browser chat for Neral 350M; never uploads local weights.

The --tool-only mode previews the UI without pretending that a model is running.
Native generation requires the locked tokenizer, matching local checkpoint and
CUDA BF16. LAN access is opt-in and password protected; share links are disabled.
"""
from __future__ import annotations

import argparse
import getpass

from chat350_web_core import run_turn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="Trusted local Neral 350M .pt file")
    ap.add_argument("--tokenizer", help="Matching locked local tokenizer.json")
    ap.add_argument("--tool-only", action="store_true",
                    help="Preview with deterministic math tools; no model inference")
    ap.add_argument("--lan", action="store_true",
                    help="Bind LAN interface for phone access (password required)")
    ap.add_argument("--port", type=int, default=9681)
    ap.add_argument("--tool-mode", choices=["auto", "off"], default="auto")
    ap.add_argument("--knowledge-dir", help="Opt-in local .txt/.md knowledge directory (read-only)")
    ap.add_argument("--knowledge-limit", type=int, default=2,
                    help="Maximum cited local passages per question (1-10)")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--max-context", type=int, default=1024)
    args = ap.parse_args()
    if not 1 <= args.knowledge_limit <= 10:
        ap.error("--knowledge-limit must be between 1 and 10")
    if not 1024 <= args.port <= 65535:
        ap.error("--port must be 1024..65535")
    if not 1 <= args.max_new < args.max_context <= 8192:
        ap.error("invalid context or generation budget")
    if not args.tool_only and not (args.checkpoint and args.tokenizer):
        ap.error("native mode requires both --checkpoint and --tokenizer; "
                 "use --tool-only for a clearly labeled interface preview")
    if args.tool_only and (args.checkpoint or args.tokenizer):
        ap.error("--tool-only cannot be combined with checkpoint paths")
    if args.tool_only and args.tool_mode == "off":
        ap.error("tool-only mode requires --tool-mode auto")

    import gradio as gr

    knowledge = None
    if args.knowledge_dir:
        from chat350_knowledge import KnowledgeIndex
        knowledge = KnowledgeIndex.from_directory(args.knowledge_dir)

    native = None
    if not args.tool_only:
        # Delay importing PyTorch/model code for the CPU-only interface preview.
        from pathlib import Path
        from chat_cli350 import generate, load_runtime
        model, tokenizer, device = load_runtime(Path(args.checkpoint),
                                                Path(args.tokenizer))

        def native(pending):
            return generate(model, tokenizer, pending, max_new=args.max_new,
                            max_context=args.max_context, device=device)

    auth = None
    host = "127.0.0.1"
    if args.lan:
        password = getpass.getpass("LAN-Chat-Passwort (mindestens 12 Zeichen): ")
        if len(password) < 12:
            ap.error("LAN password must contain at least 12 characters")
        host = "0.0.0.0"
        auth = ("neral", password)

    mode_label = ("Nur lokale Werkzeuge, kein Sprachmodell" if args.tool_only
                  else "Eigenes Neral-350M-Modell (experimentell)")
    with gr.Blocks(title="Neral 350M | Lokaler Chat") as demo:
        gr.Markdown("# Neral / Helios 350M")
        gr.Markdown(f"**{mode_label}** · lokal · keine Cloud-Modellinferenz · "
                    "nicht für zuverlässige Fachauskünfte freigegeben")
        if knowledge is not None:
            gr.Markdown(f"Lokale Wissensbasis: {knowledge.passage_count} indexierte Textabschnitte. "
                        "Quellen werden direkt angezeigt, nicht vom Modell zusammengefasst.")
        chat = gr.Chatbot(label="Gespräch", type="messages", height=470)
        state = gr.State([])
        message = gr.Textbox(label="Nachricht", placeholder="Stelle Neral eine Frage...",
                             max_lines=4, max_length=2000)
        with gr.Row():
            send = gr.Button("Senden", variant="primary")
            clear = gr.Button("Verlauf löschen")

        def respond(question, displayed, context):
            try:
                answer, source, updated = run_turn(
                    question, context or [], native=native,
                    enable_tools=args.tool_mode == "auto",
                    knowledge=knowledge, knowledge_limit=args.knowledge_limit)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            source_name = ("Neral 350M (nativ)" if source == "native-model"
                           else "Kein Modell geladen" if source == "no-model"
                           else "Lokale Dokumente (zitierter Textauszug)" if source == "tool:local-knowledge"
                           else "Deterministisches Rechenwerkzeug: " + source[5:])
            rows = list(displayed or [])
            rows.extend([{"role": "user", "content": question.strip()},
                         {"role": "assistant", "content":
                          answer + "\n\n*Quelle: " + source_name + "*"}])
            return "", rows, updated

        send.click(respond, [message, chat, state], [message, chat, state],
                   api_name=False, concurrency_limit=1)
        message.submit(respond, [message, chat, state], [message, chat, state],
                       api_name=False, concurrency_limit=1)
        clear.click(lambda: ([], []), outputs=[chat, state], api_name=False)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=host, server_port=args.port, share=False, auth=auth,
        inbrowser=False, show_error=False)


if __name__ == "__main__":
    main()

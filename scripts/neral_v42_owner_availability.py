"""Read-only 9690 owner availability probe using the established cookie+CSRF session.

No prompts, model inference, datasets, private answer keys or model weights.
"""
from __future__ import annotations

from scripts.neral_v41_blind200_live import OwnerClient

STATUS = "/api/neral/chat/three-tier/status"
AVAILABILITY = "/api/neral/chat/three-tier/availability"
EXPECTED = {"Amber": 25, "Quartz": 40, "Obsidian": 70}


def verify_owner_available(client=None):
    client = OwnerClient() if client is None else client
    client.connect()  # obtains owner-local HTTP session and CSRF cookie
    identity = client.get(STATUS)
    state = client.get(AVAILABILITY)
    if (identity.get("enabled") is not True or
            identity.get("public_tiers") != list(EXPECTED) or
            state.get("busy") is not False or
            state.get("deadline_seconds") != EXPECTED):
        raise RuntimeError("NERAL session, service identity, request timer or GPU lock mismatch")
    return state


def main():
    verify_owner_available()
    print("NERAL_V42_AUTHENTICATED_OWNER_IDLE_25_40_70_HTTP200", flush=True)


if __name__ == "__main__":
    main()

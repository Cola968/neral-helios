# NERAL / Helios — targeted security review (2026-11-06)

Scope: checked-in Flask app, LAN gateway, billing helpers, CI and Git history.
This is a source audit with focused tests, **not** a penetration test.
No payments, token rotations or revocations were made.

## 1. Authorization and user data

In app.py the global before-request guard checks loopback address, Host,
Origin, Sec-Fetch-Site, and a process-generated session cookie or trusted
gateway header for every /api/ request.

| Route | Data or operation | Verified control | Ownership |
| --- | --- | --- | --- |
| GET/POST /api/projects | List/create local projects | Central session guard | Single installation |
| GET /api/projects/<project> | Read project graph | Central guard | No per-user owner |
| POST /api/projects/<project>/<action> | Modify/review/undo project | Guard, action allowlist, CASCADE confirmation | No per-user owner |
| GET /api/chats | Read synchronized chats | Central guard | No per-user owner |
| POST /api/chats/<key> | Write/tombstone chat | Guard, revision checks | No per-user owner |
| POST /api/generate | Prompt/history/inference | Guard and size limits | Shared runtime |
| POST /api/cancel | Cancel inference job | Guard | Shared jobs |
| POST /api/model | Select named checkpoint | Guard and allowlist | Shared runtime |
| GET /api/info, /api/metrics, /api/health | Runtime metadata | Guard | Shared runtime |

GET / issues the install-wide local cookie. phone_gateway.py separately
requires pairing and forwards to the same local owner. This application is
**single-owner, not multi-tenant**. Paired devices can access the owner's
shared data. No individualized user identity, resource-owner field or
admin role currently exists. This is a deployment limitation, not evidence
that cross-user access is safe. Internet or multiuser deployment requires
identity-aware server-side authorization for every read/change/delete,
including inference jobs. Existing tests cover selected guards in
tests/test_projects.py, tests/test_helios_workspace.py and
tests/test_helios_install.py. The new route-inventory test is a change gate,
not a proof of authorization correctness.

## 2. Secrets and credentials

.gitignore excludes local_state/ and .env* except examples. setup_phone.py
writes a private TLS key, pairing instructions and pairing config.
Previously default POSIX permissions could leave directories at 0755 and
credentials at 0644 under a permissive umask. It also accepted 0.0.0.0 and
unsupported network addresses. The patch creates new credential files
exclusively, uses 0700 directories and 0600 files on POSIX, and refuses
unspecified, link-local or IPv6 addresses. Existing credential files are
not automatically modified.

GitHub Secret Scanning and Code Scanning were disabled at review time.
A new redacted, high-signal scanner examines tracked files and reachable
Git patch history. It outputs only paths, line numbers, category and short
commit identifiers, never credential values. It is NOT comprehensive:
binary files, workflow logs, build artifacts, full secret entropy detection
and pre-existing Windows ACLs need separate checks. Enabling native GitHub
scanning and a dedicated full scan remains advisable. Actual leaked secrets
must be assessed and rotated only with explicit operator approval.

Windows NTFS ACL inheritance is not proven secure by POSIX chmod. Inspect
existing and new pairing files on the user's Windows machine before wider
mobile deployment.

## 3. Payments and subscription access

billing/entitlements.py provides a plan-to-feature mapping only.
billing/README.md describes Stripe as prepared, not live. No Checkout,
Customer Portal or Stripe webhook endpoint was found in the reviewed HTTP
routes, so there was no payment flow to exploit or test.

The helper accepting a caller-provided plan must never be treated as proof
of payment. Future paid access must be derived from a server-side,
user-bound subscription, with allowlisted price mapping, raw-body webhook
signature verification, persistent event deduplication, expiry/cancellation
handling and failed-payment policy. Client values or checkout return URLs
cannot grant privileges. No real payments or billing settings were changed.

## 4. File uploads

No HTTP upload, multipart or request.files handler was found in the reviewed
app. Local model-file operations are not a public upload API. Both app and
LAN gateway cap request bodies at 256 KiB.

No fictional upload endpoint was added. Before adding one, enforce limits
on size and count, validate types from content, randomize names, store files
outside the served root, block traversal/symlinks/overwrite, require owner
checks, and add tests for unauthorized access and rejected files.

## Tests and unresolved verification

- Eight focused phone-setup tests passed locally after reproducing original
  permission and address-validation failures.
- Added synthetic tests for token detection, Git history detection and
  non-disclosure of literal values.
- Added a static HTTP route inventory to force review when API or billing
  routes change. This does not prove live endpoint security.
- Extended security-regression CI with separate lightweight checks and the
  redacted Git-history scan. Check GitHub results before merging.
- Manual work still needed: real Windows file ACL checks, live LAN/TLS
  pairing and browser tests, full secrets scanning across artifacts/logs,
  and eventual multiuser/billing/real upload integration testing.

**Do not interpret this audit as approval to expose NERAL publicly.**

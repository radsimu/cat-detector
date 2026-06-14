# ADR-0004: Authentication Approach

**Status:** Accepted  
**Date:** 2026-04-06  
**Author:** Cat Detector Team

---

## Context

The service exposes:

1. A **web UI** for browsing detections and triggering fine-tuning
2. A **REST API** consumed by that UI
3. An **outbound webhook** sent to an external endpoint

We need to decide how to protect each of these surfaces. The threat model is a home network deployment — the service is not intended to be exposed to the public internet, but it may be accessible on a local network shared with other household members or devices.

This ADR also covers the outbound webhook secret, which is a distinct authentication concern (we are the caller, not the verifier).

---

## Options Considered

### Inbound: Web UI + API access

| Option | Pros | Cons |
|---|---|---|
| **No authentication (rely on network boundary)** | Zero implementation cost; no credentials to manage | Anyone on the LAN can access the UI and trigger fine-tuning or read detection frames |
| HTTP Basic Auth (single shared password) | Simple; supported natively by browsers and `httpx` | Credentials in plaintext unless served over HTTPS; no per-user audit trail |
| API key in header (`X-API-Key`) | Stateless; easy to revoke | Requires HTTPS to be safe; UI must store and send the key |
| JWT / OAuth2 | Industry standard; stateless tokens; supports scopes | Significant implementation complexity; no identity provider in a home setup |
| Reverse proxy auth (nginx `auth_basic`, Authelia, Tailscale) | Pushes auth out of the application; reusable across services | Requires additional infrastructure the user may not have |

### Outbound: Webhook secret

| Option | Pros | Cons |
|---|---|---|
| **Shared secret in `X-Webhook-Secret` header** | Simple; receiver can validate with a string comparison | Secret sent on every request — requires HTTPS on the receiver side to be safe |
| HMAC signature (like GitHub webhooks) | Cryptographically verifiable; replay-resistant with a timestamp | More implementation complexity; receiver must implement HMAC verification |
| No secret | Zero setup | Receiver cannot verify the request came from this service |

---

## Decision

### Inbound authentication: None in the application layer

The service ships with **no built-in authentication**. Access control is delegated to the network layer:

- **Recommended deployment:** bind to `127.0.0.1` (localhost only) or behind a Tailscale/VPN tunnel
- **If LAN exposure is needed:** place behind a reverse proxy (nginx, Caddy) with HTTP Basic Auth or Tailscale's built-in auth
- The `HOST` config defaults to `0.0.0.0` to support Docker and home server deployments, but the README explicitly warns that this should be behind a firewall

Rationale: the labeling UI and fine-tuning trigger are low-stakes operations in a home context. The cost of implementing and maintaining auth (credential rotation, browser storage, HTTPS setup) outweighs the risk for a training project with a single expected user. This decision is explicitly revisited if the service is ever exposed to the internet.

### Outbound webhook: Optional shared secret

An optional `WEBHOOK_SECRET` environment variable can be set. When present, it is sent as an `X-Webhook-Secret` header with every webhook POST. The receiving endpoint is responsible for validating it.

HMAC signing was considered but rejected: it requires the receiver to implement a specific verification algorithm, which adds friction when testing with simple webhook endpoints (webhook.site, ntfy.sh, etc.). A plain shared secret is sufficient for a home-use webhook where HTTPS on the receiver is a reasonable assumption.

---

## Consequences

**Positive:**
- Zero authentication code to write, test, or maintain
- No credential management burden for the user
- Webhook secret is opt-in — the service works without it, which simplifies initial setup and testing
- Deploying behind Tailscale gives strong authentication with zero application code

**Negative:**
- Any device on the same network as the service can access the UI, trigger fine-tuning, and download detection frames. This is a deliberate and documented trade-off, not an oversight
- The `X-Webhook-Secret` header provides no replay protection — a captured request can be replayed. Acceptable for a cat alert; not acceptable for financial or security-critical webhooks
- If the threat model changes (e.g. the service is port-forwarded), the entire authentication approach must be revisited. There is no incremental hardening path within the current codebase

**Revisit triggers:**
- Service is exposed beyond the local network or Tailscale mesh
- Multiple users with different permission levels need access (e.g. a family member can view but not trigger training)
- Detection frames contain sensitive information (e.g. the camera captures more than the sofa)

**Recommended next step if auth becomes necessary:**
Add HTTP Basic Auth via a FastAPI middleware or a Caddy reverse proxy — both require under 10 lines of configuration and no changes to application logic.

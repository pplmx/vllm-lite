# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅                 |

## Reporting a Vulnerability

If you discover a security vulnerability, please open a **GitHub Security Advisory** or contact maintainers directly.

Please include the following information:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (branch/commit)
- Any special configuration required to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Response Timeline

We aim to acknowledge vulnerability reports within **48 hours** and provide a more detailed response within **7 days**. We will work with the reporter to understand and resolve the issue, and keep them updated on our progress.

## Security Best Practices

- **API Keys**: Always use strong API keys in production
- **Rate Limiting**: Configure appropriate rate limits for your use case
- **Network**: Run behind a reverse proxy (nginx) for TLS termination
- **Updates**: Keep vLLM-lite updated to the latest version

## Scope

The following are **in scope** for security fixes:

- Authentication bypass
- Rate limiting bypass
- Memory safety issues
- Input validation vulnerabilities

The following are **out of scope** (handled externally):

- TLS/SSL → Configure at load balancer/reverse proxy
- DDoS protection → Use CDN or cloud DDoS protection
- Audit logging → Integrate with external SIEM

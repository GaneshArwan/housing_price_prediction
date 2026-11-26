# Project Completion & Security Protocol

Before considering any task in this repository complete, verify and report this checklist:

- [ ] **Security Skill Loop**: Run 5 security checks (Bandit SAST, secret/credential scan, path traversal & deserialization & input validation audit, Dockerfile & CI hardening check, guardrails coverage check). Fix any issues and re-run until 100% pass.
- [ ] **Dependency Audit**: Check Python/NPM dependencies for known high/critical vulnerabilities if applicable.
- [ ] **Professional Language Formatting**: Ensure user-visible text and documentation contain no AI-buzzword filler; use direct, factual wording.
- [ ] **Guardrails**: Maintain and verify Pydantic / finite-number prediction guardrails (`src/guardrails.py`, `src/predict.py`, `app.py`).

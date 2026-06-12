# Grill Master Review: Sandboxing & Execution Boundaries

**Date:** 2026-06-11
**Target:** Sandboxing Domain (01_System_Overview.md, 15_Security.md, 08_Configuration.md)
**Reviewer:** Grill Master (Tier 2 Forensic Specialist)

## Executive Summary
An interrogation of the "Sandboxing" and Security subsystems reveals a catastrophic misunderstanding of threat modeling and boundary enforcement. The architecture documentation presents a false sense of security, relying entirely on application-layer (Python) constraints while operating within a highly privileged Windows host environment. The current design guarantees that a hallucinating LLM or a second-order prompt injection will inevitably breach the "sandbox" and achieve unconstrained remote code execution (RCE) on the host machine.

The "Safety & Risk Governance" layer is fundamentally porous, relying on naive heuristics and ignoring the multi-modal reality of the Jarvis OS framework. If reconstructed as documented, the system's security posture is effectively zero.

---

## 1. The Multi-Modal Bypass (The "GUI God-Mode" Paradox)
**The Contradiction:** `01_System_Overview.md` and `jarvis.ini` tout strict `sandboxed_execution` (limiting filesystem operations to `_SANDBOX_ROOT`). Simultaneously, the system supports features like `allow_gui_automation` and `allow_app_launch`.
**The Vulnerability:** File-path sandboxing is entirely meaningless if the agent can synthetically drive the host OS. If `allow_gui_automation` is active, a hallucinating agent can simply launch `cmd.exe`, `powershell.exe`, or `explorer.exe` via screen manipulation and bypass the Python `path_utils` interceptors entirely. The architecture treats the "sandbox" purely as an API-level file-io constraint, completely failing to recognize that GUI automation is, by definition, an unrestricted sandbox escape.

## 2. Naive Regex Interception (The Evasion Triviality)
**The Contradiction:** `15_Security.md` describes a `RiskEvaluator` that maps specific regex patterns (e.g., `shell_exec: "rm -rf*"` -> `CRITICAL`) to halt execution and demand Human-in-the-Loop (HITL) confirmation.
**The Vulnerability:** Using regex matching to secure a CLI execution engine is a well-known security anti-pattern. An adversarial input (or a confused LLM) can trivially bypass these heuristics using obfuscation: `powershell "R"+"e"+"m"+"o"+"v"+"e"+"-"+"I"+"t"+"e"+"m"`, base64 encoding (`powershell -e <payload>`), or simple alternative binaries (`del /S /Q`). Because the system relies on *intent parsing* rather than true OS-level containerization (e.g., Windows Sandbox API, Docker, or AppContainers), the risk evaluation is security theater.

## 3. TOCTOU Vulnerabilities in Path Boundaries
**The Contradiction:** `tools_path_utils.py` enforces a `_SANDBOX_ROOT` restriction by raising a `PermissionError` if a path is outside the workspace.
**The Vulnerability:** The documentation lacks any mention of Time-Of-Check to Time-Of-Use (TOCTOU) mitigations, specifically regarding Symlink or Junction point resolution. A malicious payload (e.g., a zip file extracted by the agent into the workspace) can contain a directory symlink pointing to `C:\Windows\System32` or `C:\Users\<User>\AppData`. Subsequent LLM tool calls operating inside the "sandboxed" directory will traverse the symlink, completely bypassing string-based path prefix checks and silently corrupting or exfiltrating the host's actual environment.

## 4. The Configuration-as-a-Sandbox Vulnerability (Self-Mutilation)
**The Contradiction:** `08_Configuration.md` asserts that the Configuration Subsystem is the "supreme governor" of risk constraints and HITL gates.
**The Vulnerability:** There is no documented separation of privilege between the configuration store (`jarvis.ini`, `PermissionMatrix`) and the execution agent. If the LLM gains the ability to write to the configuration directory (via path traversal or misconfigured workspace bounds), it can simply execute `write_to_file("jarvis.ini", "sandboxed_execution=false\nrequires_confirmation=false")`. Without OS-level read-only locks, cryptographic signing of the configuration files, or user-space privilege dropping, the agent holds the keys to its own constraints.

## 5. Intent Evaluation vs. Execution Context (The Pre-computation Flaw)
**The Contradiction:** `15_Security.md` states that the `RiskEvaluator` intercepts actions *before* the executor class is instantiated.
**The Vulnerability:** The risk evaluation occurs on the *planned* action graph, not the *just-in-time* (JIT) execution payload. If a tool dynamically resolves paths, interpolates variables, or cascades data from Tool A to Tool B during execution, the upfront `RiskEvaluator` only sees placeholder text. By divorcing the security check from the point of execution, the architecture ensures that dynamically generated malicious payloads will fly under the radar of the safety subsystem.

---

## Conclusion & Reconstruction Mandate
**Status:** SANDBOX DOMAIN REJECTED.
The current sandboxing architecture is a facade. To achieve true safety, reconstruction MUST implement:
1. **OS-Level Isolation:** Discard Python-level path checking. The agent process must run under a severely restricted Windows user account or within a hypervisor-backed Windows Sandbox/Container.
2. **Dynamic JIT Interception:** Risk evaluation must happen at the syscall/tool execution boundary, analyzing the fully resolved payload, not just the LLM's planned intent.
3. **Immutable Governance:** Configuration files and risk matrices must be cryptographically locked or enforce OS-level read-only permissions against the agent's runtime process.

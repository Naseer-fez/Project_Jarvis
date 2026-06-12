# Adversarial Critique: Networking Domain

## 1. The Egress SSRF Vulnerability (Server-Side Request Forgery)
**The Flaw:** The architecture abstracts network operations in `11_APIs.md` and provides strict `RiskEvaluator` gates in `15_Security.md`, but it completely fails to implement an **Egress Network Filter**.
**Devastating Impact:** The system is fundamentally vulnerable to SSRF. An attacker can use a prompt injection (e.g., via a seemingly benign web page Jarvis is asked to summarize) to coerce the Web or API integration tool to fetch `http://169.254.169.254` (cloud metadata) or local administrative endpoints (e.g., `http://localhost:11434` for Ollama manipulation). Because the agent acts as the network origin, it bypasses the `AuthManager` and CSRF protections entirely. Without a strict CIDR denylist for outbound `aiohttp` / `urllib` requests, Jarvis can be weaponized against its own host network.

## 2. ThreadPool Exhaustion via Synchronous Network SDKs
**The Flaw:** `11_APIs.md` proudly claims to wrap synchronous third-party SDKs (like Twilio, GitHub, Spotify) in `asyncio.get_event_loop().run_in_executor()`.
**Devastating Impact:** Python's default `ThreadPoolExecutor` has a finite number of worker threads (defaulting to `min(32, os.cpu_count() + 4)`). If a network outage occurs or a third-party API begins dropping packets (TCP blackhole), these synchronous calls will block indefinitely (or until the massive 5-minute agent loop timeout fires). If an agent plan spams 40 tool calls to a degraded endpoint, the *entire thread pool will saturate*. Once the pool is exhausted, *no other synchronous background tasks can execute*, creating an insidious deadlock that freezes the OS without crashing the event loop.

## 3. The "Thundering Herd" API Rate Limit Cascade
**The Flaw:** `14_Error_Handling.md` implements a naive LIFO retry mechanism with exponential backoff on the *execution graph step*. 
**Devastating Impact:** This is a localized retry logic, not a global one. If a multi-step Plan DAG spawns 5 concurrent tool calls to the same endpoint (e.g., Notion or GitHub API), and the endpoint returns an HTTP 429 Rate Limit, *all 5 nodes will independently back off and retry simultaneously*, exacerbating the rate limit. Without a centralized Token Bucket, Circuit Breaker, or global API rate-limit semaphore per external domain, Jarvis will continuously DDOS itself and trigger permanent vendor bans during parallel tool execution.

## 4. Unmitigated DNS Blocking & Async Freeze
**The Flaw:** The system relies heavily on `aiohttp` for async HTTP calls, enforcing application-level `TIMEOUT_S`.
**Devastating Impact:** In standard Python setups, `asyncio`'s DNS resolution uses the synchronous `getaddrinfo()` under the hood, running in the default thread pool. If the host network's DNS server is slow or hijacked, the DNS resolution blocks the worker threads. Application-level `aiohttp` timeouts *do not interrupt blocking DNS lookups*. This will cause the entire asynchronous event loop to stall helplessly before the HTTP request even begins. 

## 5. WebSocket Backpressure and OOM Collapses
**The Flaw:** The `Event Bus` streams `LLM_STREAM_CHUNK` and state changes directly to the `DashboardRuntime` via WebSockets for real-time UI updates.
**Devastating Impact:** The system lacks backpressure management on its WebSockets. If the LLM Orchestrator is utilizing a high-speed inference engine (like Groq outputting 800 tokens/sec), but the client connection is dropping packets or experiencing high latency, the WebSocket buffer on the server will bloat exponentially. The Event Bus will continue stuffing tokens into memory until the server hits an Out-Of-Memory (OOM) exception, tearing down the entire Jarvis process simply because a dashboard tab lost signal.

## Conclusion
The Networking and API boundary in Jarvis acts as a naive pass-through wrapper rather than a hardened perimeter. It assumes external networks are fast, honest, and resilient. To fix this, the architecture requires an Egress IP Firewall, Global API Circuit Breakers, explicit `aiodns` implementation, and WebSocket high-water marks.

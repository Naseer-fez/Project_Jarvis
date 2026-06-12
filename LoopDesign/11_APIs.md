# 11 APIs (Clients & Integrations)

## 1. System Intent
Jarvis is an autonomous AI agent that must interpret instructions, reason about state, and interact with the physical and digital world. Without a normalized API abstraction layer, the core reasoning loop would be tightly coupled to specific third-party REST protocols, unstable vendor endpoints, and varying LLM payload formats. The APIs subsystem (often referred to as Clients or Integrations) exists to provide clean, unified interfaces over these external complexities. It insulates Jarvis from the volatility of the outside world, ensuring that failures in a single cloud endpoint or external service do not corrupt or crash the core operational loop.

## 2. Core Responsibilities
The APIs subsystem fundamentally owns the boundaries of the system, managing both inbound intelligence (Models) and outbound actuation (Integrations).

**Model API Responsibilities (`core/llm/`):**
- **Unified Abstraction:** Normalizing multi-provider LLM requests across local (Ollama) and cloud (Gemini, Groq, OpenAI, Anthropic) instances into standard `complete()` interfaces.
- **Intelligent Routing:** Handling tiered model selection (e.g., Tier 1 for fast tasks like query extraction, Tier 3 for complex reasoning tasks) and fallback execution if a primary endpoint fails.
- **Payload Management:** Parsing, cleaning, and formatting responses (e.g., stripping `<think>` tokens emitted by models like DeepSeek).
- **Telemetry Tracking:** Recording token usage, latency, and success/failure rates for system observability.

**Integration API Responsibilities (`integrations/clients/`):**
- **Service Wrapping:** Abstracting complex third-party APIs (e.g., Twilio for WhatsApp, Open-Meteo for Weather, GitHub, Notion, Home Assistant, Spotify) into simplified local Python calls.
- **Execution Safety:** Enforcing strict timeouts (`TIMEOUT_S`) on all network operations to prevent indefinite stalls.
- **Async Bridging:** Wrapping synchronous SDKs or blocking HTTP requests in `asyncio.get_event_loop().run_in_executor()` to ensure the agent loop is never blocked.
- **Data Normalization:** Converting raw JSON or complex REST responses into clean, summarized dataclasses (e.g., `SearchResult`) suitable for ingestion into an LLM's context window.

## 3. System Interactions
- **Agent Loop & Controller:** The `ModelRouter` is injected into the core loop. When the system requires a completion, it calls the routing layer. The router attempts local inference (`OllamaClient`) or directs the request to the `CloudLLMClient` based on configuration and tier requirements.
- **Tool Registry:** The `BaseIntegration` subclasses are exposed to the AI agents through specific tools (e.g., `web_tools`, `whatsapp`). When an agent decides to take an action, it invokes the tool, which internally executes the API wrapper.
- **External Web & Providers:** The subsystem communicates outbound via `aiohttp` or `urllib.request` to REST endpoints. It manages the serialization of intents to HTTP payloads and deserialization of HTTP responses to agent context.

## 4. Failure Impact
- **Loss of Intelligence:** If the Model APIs are removed or broken, the `ControllerV2` is entirely paralyzed. Jarvis would lose all capacity to generate responses, plan actions, or understand context.
- **Loss of Agency:** If the Integration APIs are removed, Jarvis reverts to a disconnected, "brain-in-a-vat" chatbot. It loses its defining autonomous capabilities—it can no longer control the computer, fetch web data, manage calendars, or send messages.
- **System Instability:** Without the abstraction layer's strict timeouts and async wrappers, a single rate-limit error or network latency spike from a 3rd-party provider would stall the event loop, causing the entire system to hang and requiring a hard restart.

## 5. Reconstruction Guide
If this subsystem must be rebuilt from scratch without source code, follow these steps:

1. **Establish the Model Layer:**
   - Define a `BaseLLMClient` protocol with a standard `async def complete(...)` interface.
   - **Local Client:** Implement an `OllamaClient` that sends a JSON payload `{prompt, system, stream: False}` to `http://localhost:11434/api/generate`. Parse the `response` field.
   - **Cloud Client:** Implement a `CloudLLMClient` mapping a tiering logic (e.g., 1=fast, 2=balanced, 3=capable) to specific vendor endpoints (Groq, Gemini, Anthropic, OpenAI). Handle vendor-specific authentication and response extraction.
   - **Model Router:** Wrap both clients. Implement a retry mechanism (e.g., 3 retries on `ClientError`) and a fallback chain (if local fails, failover to cloud; if provider A fails, failover to provider B).

2. **Establish the Integrations Layer:**
   - Create a `BaseIntegration` abstract base class.
   - For each service (Weather, WhatsApp, etc.), implement a subclass that handles the specific authentication (e.g., API keys loaded from `jarvis.ini` or environment variables).
   - Use `aiohttp` for native async endpoints. For SDKs that only support synchronous execution (like Twilio), aggressively wrap calls in thread pool executors.
   - Always map the raw JSON payload back to plain text or a simple, typed dataclass before returning it, ensuring the LLM receives highly dense, readable context.

## 6. Exact Programmatic Schemas (Blind Rebuild Assets)

### 6.1. Core Interface Signatures & Timeouts
**LLM Client Interface:**
```python
async def complete(self, prompt: str, system: str='', temperature: float=0.1, task_type: str='chat', keep_think: bool=False, classification: dict[str, Any] | None=None) -> str
```
**Integration Interface & Standard Return Type:**
```python
@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""

# BaseIntegration requirements
def is_available(self) -> bool: ...
def get_tools(self) -> list[dict[str, Any]]: ...
async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]: ...
```
**Timeouts:** LLM Provider network timeout is `45` seconds. Integration network timeout is `10` seconds.

### 6.2. LLM Provider Payload Mappings (DTOs)

**OpenAI / Groq Vendor Schema:**
```json
{
  "model": "<model_name>",
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user", "content": "<prompt>"}
  ],
  "temperature": <temperature>,
  "max_tokens": 2048
}
```
*Extraction Path:* `data["choices"][0]["message"]["content"]`
*Usage Extraction:* `data["usage"]["prompt_tokens"]`, `data["usage"]["completion_tokens"]`

**Anthropic Vendor Schema:**
```json
{
  "model": "<model_name>",
  "max_tokens": 2048,
  "system": "<system_prompt>",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "temperature": <temperature>
}
```
*Required Headers:* `"x-api-key"`, `"anthropic-version": "2023-06-01"`
*Extraction Path:* `data["content"][0]["text"]`

**Gemini Vendor Schema:**
```json
{
  "contents": [{"parts": [{"text": "<prompt>"}]}],
  "generationConfig": {"temperature": <temperature>},
  "systemInstruction": {"parts": [{"text": "<system_prompt>"}]}
}
```
*Extraction Path:* `data["candidates"][0]["content"]["parts"][0]["text"]`

### 6.3. Integration Tool Signatures & Normalization Models

**Configuration Keys:**
- `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM`
- `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`
- `GITHUB_TOKEN`, `GITHUB_DEFAULT_REPO`

**Weather (`get_current_weather`):**
- **Tool Schema Args:** `{"city": {"type": "string"}}`
- **Normalized Return Schema:**
  ```json
  {
    "city": "string",
    "country": "string",
    "temperature_c": "float",
    "humidity": "float",
    "wind_speed_kmh": "float"
  }
  ```

**WhatsApp (`send_whatsapp`):**
- **Tool Schema Args:** `{"to": {"type": "string"}, "message": {"type": "string"}}`
- **Normalized Return Schema:** `{"sid": "string"}`

**GitHub (`list_open_issues`):**
- **Tool Schema Args:** `{"repo": {"type": "string"}, "label": {"type": "string"}, "assignee": {"type": "string"}, "milestone": {"type": "string"}, "limit": {"type": "integer"}}`
- **Normalized Return Schema:**
  ```json
  {
    "repo": "string",
    "issues": [
      {
        "number": "integer",
        "title": "string",
        "state": "string",
        "url": "string",
        "labels": ["string"]
      }
    ]
  }
  ```

**Home Assistant (`get_entity_state`):**
- **Tool Schema Args:** `{"entity_id": {"type": "string"}}`
- **Normalized Return Schema:**
  ```json
  {
    "entity_id": "string",
    "state": "string",
    "attributes": {"dict": "Any"}
  }
  ```

**Notion (`create_page`):**
- **Tool Schema Args:** `{"parent_id": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"}, "parent_type": {"type": "string"}}`
- **Normalized Return Schema:**
  ```json
  {
    "id": "string",
    "url": "string"
  }
  ```

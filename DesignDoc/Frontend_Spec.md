# Frontend Specification

## Architecture Overview

The Jarvis frontend is a server-rendered web dashboard built using **FastAPI** with **Jinja2Templates** for HTML rendering and a vanilla JavaScript application for interactivity. The layout relies on standard HTML/CSS, avoiding heavy frameworks, and integrates a custom WebSocket-based reactive state management system.

### Layouts
- **Base Layout (`base.html`)**: 
  - **Sidebar Navigation**: Contains links to all major modules (Command, Memory, Goals, AI OS, Search, Converter, Auto-Clicker, Health). Highlights the active route.
  - **Particle Background**: A canvas-based interactive background (`ParticleCanvas`) that connects floating nodes and reacts to mouse movement.
  - **Main Content Area**: Injected via Jinja2 blocks (`{% block content %}`).
  - **Global Status**: Contains a visual WebSocket connection indicator in the footer.
  - **Global Scripts**: Loads `app.js` providing core functionality (WebSockets, Toast system, counters).

### State Management
- **WebSocket (`/ws`)**: A live feed pushing Jarvis's internal state to the frontend every 2 seconds. The payload includes `state`, `last_response`, `last_input`, `session_id`, `memory_count`, `active_goals`, `ollama_online`, `uptime_seconds`, and `model`.
- **`app.js` (Client-Side)**: 
  - Uses the `JarvisWebSocket` class to automatically connect and handle reconnects.
  - The `renderState(data)` function reacts to the WebSocket payload by directly mutating DOM elements across the dashboard (e.g., updating the State Orb color, updating animated counters, setting the offline/online status).
- **Backend State (`JarvisState`)**: A singleton data class in `server.py` wrapped by a threading lock, updated asynchronously by core systems.

### Routing
Routing is handled server-side by **FastAPI** `server.py` utilizing cookie-based sessions (`jarvis_session`) and an API token fallback (`X-Dashboard-Token`).
- **`/login` (GET/POST)**: Login UI and authentication.
- **`/logout` (GET)**: Clears cookies and redirects.
- **`/` (GET)**: Command Center (Index).
- **`/memory` (GET)**: Memory Browser.
- **`/goals` (GET)**: Goals Manager.
- **`/ai-os` (GET)**: AI OS Overview.
- **`/search` (GET)**: Fast Search interface.
- **`/converter` (GET)**: Universal Converter.
- **`/clicker` (GET)**: Auto-Clicker configuration.
- **`/health-ui` (GET)**: System Health dashboard.

---

## Page Specifications

### 1. Login Page (`/login`)
- **Purpose**: Authenticates users before allowing access to the dashboard system.
- **Components**: Login Card, Logo SVG, Error Notification, Authentication Form.
- **Inputs**: Username (`#username`), Password (`#password`).
- **User Actions**: Submit credentials.
- **Data Sources**: Submits a `POST` request to `/login`, which checks against the local SQLite `auth.db`.
- **Navigation Paths**: 
  - Success: Redirects to `/` (Command Center).
  - Failure: Re-renders `/login` with an error template string.

### 2. Command Center (`/`)
- **Purpose**: The primary home dashboard to monitor core state and send direct commands to the Jarvis agent.
- **Components**: 
  - **State Orb**: Visual CSS-animated orb indicating Jarvis's status (e.g., IDLE, PROCESSING, OFFLINE).
  - **Stats Grid**: Displays active Session ID, Model, Memory count, Active Goals count, Ollama connectivity, and system Uptime.
  - **Command Input**: Form to capture user text commands.
  - **Response Panel**: Displays the latest command input string and the text response.
- **Inputs**: Text Command (`#command-input`). Keyboard shortcut `Ctrl+K` binds to focus.
- **User Actions**: Submit command form.
- **Data Sources**: 
  - Live stats stream from WebSocket `/ws`. 
  - Sends a `POST` request to `/command` which pipes instructions to the core Jarvis controller loop.
- **Navigation Paths**: Sidebar navigation links to all other modules.

### 3. Memory Browser (`/memory`)
- **Purpose**: Interface to query and read Jarvis's persisted AI memory data (preferences, episodes, conversations).
- **Components**: Page header, Search query form, Empty state graphic, Memory Cards Grid (categorized by memory type).
- **Inputs**: Search string `q` (GET parameter).
- **User Actions**: Submit search form.
- **Data Sources**: Uses `sqlite3` to query the `jarvis_memory.db` database directly via backend route, passing `memories` context to Jinja2.
- **Navigation Paths**: Form submission reloads the page with `?q={query}`.

### 4. Goals Manager (`/goals`)
- **Purpose**: Track and manage Jarvis's priority objectives.
- **Components**: Goal cards grid (displaying priority, status badge, description, timestamp, and a completion button), "Add New Goal" Form.
- **Inputs**: 
  - Description string (`#goal-description`)
  - Priority dropdown 1-10 (`#goal-priority`)
- **User Actions**: 
  - Add a new goal (Submits via `fetch` to `POST /goals/add`).
  - Complete an existing goal (Submits via `fetch` to `POST /goals/complete/{goal_id}`).
- **Data Sources**: Backed by the Jarvis `_goal_manager` which provides a list of active goals injected into the template on page load.
- **Navigation Paths**: Completing or adding a goal successfully triggers `location.reload()` to refresh the state.

### 5. AI OS Overview (`/ai-os`)
- **Purpose**: Dashboard illustrating the local-first automation platform's configuration, principles, and loaded components.
- **Components**: Hero Section, Errors alert block, Blueprint Details (Principles, Architecture Layers, Workflow Nodes, Security Controls), Plugins Table, Workflow Templates Grid.
- **Inputs**: None. Read-only view.
- **User Actions**: None.
- **Data Sources**: Populated by the `_load_ai_os_overview()` backend function which inspects the active `PluginCatalog` and AI OS blueprints.
- **Navigation Paths**: None.

### 6. Fast Search (`/search`)
- **Purpose**: A multi-threaded file search and grep interface running across the local PC.
- **Components**: Configuration Form, Stats Panel (Engine, Folders Scanned, Files Scanned, Matches, Elapsed Time), Results Data Table.
- **Inputs**: 
  - Search Path (`#search-path`)
  - Thread count select (`#search-threads`)
  - Filename Pattern (`#search-query`)
  - Content Search/Grep (`#search-content`)
  - Case Sensitive toggle (`#search-case`)
  - Include System Folders toggle (`#search-noskip`)
- **User Actions**: Run scan (POST to `/api/search`), click file links in results.
- **Data Sources**: Executes the internal `run_fast_search` tool dynamically via API. 
- **Navigation Paths**: Clicking a search result links to `/api/view-file?path={encoded_path}` to download/view the file securely.

### 7. Universal Converter (`/converter`)
- **Purpose**: Tool for converting files between various formats on demand (e.g. PNG to JPG, CSV to JSON).
- **Components**: Drag & Drop Upload Zone, File Info Details, Format Configuration, Conversion Log Panel.
- **Inputs**: 
  - File input via drag & drop or click (`#file-input`)
  - Target Format dropdown (`#target-format`)
  - Custom Extension text field (`#custom-format`)
- **User Actions**: Upload file, select format, trigger "Convert & Download", reset state.
- **Data Sources**: Client side JavaScript generates a file `FormData` object posted to `POST /api/convert`, which executes the `perform_conversion` core tool. The backend responds with a downloadable blob.
- **Navigation Paths**: The download happens entirely on-page via a dynamic `<a>` tag triggering a Blob URL download.

### 8. Auto-Clicker (`/clicker`)
- **Purpose**: Manage a vision-based screen automation script tool.
- **Components**: Status Indicator Banner, Run Stats (Attempts, Successes, Failures), Configuration form, Live Log panel, Recent Screenshots gallery.
- **Inputs**: 
  - Target Text (`#clicker-target`)
  - Interval in seconds (`#clicker-interval`)
  - Min Confidence slider/input (`#clicker-confidence`)
  - Continuous Mode toggle switch (`#clicker-continuous`)
- **User Actions**: Start Clicker, Stop Clicker, Clear Logs.
- **Data Sources**: 
  - Constantly polls `GET /api/clicker/state` and `GET /api/clicker/screenshots` using `setInterval` to refresh stats, logs, and images.
  - Submits configuration to `POST /api/clicker/start`.
- **Navigation Paths**: None. Runs entirely in the background.

### 9. System Health (`/health-ui`)
- **Purpose**: Real-time diagnostic interface showcasing dependency statuses and system health.
- **Components**: Overall Status Hero (with animated color-coded orb), Summary Stats Grid (Uptime, Checks Passed, Warnings, Failures), Diagnostic Checks Table.
- **Inputs**: None.
- **User Actions**: Click "Refresh" button to re-run diagnostics.
- **Data Sources**: Client asynchronously queries the lightweight readiness probe at `GET /health` which triggers `run_lightweight_health_check()`.
- **Navigation Paths**: None.

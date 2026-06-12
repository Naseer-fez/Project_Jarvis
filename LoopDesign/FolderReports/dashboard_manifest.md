# Dashboard Folder Manifest

## High-Level Purpose
The `dashboard` folder contains the web interface and server-side code for a web-based dashboard or frontend. It provides the user interface for interacting with various functionalities of the Jarvis AI system, utilizing a Python-based backend server (likely Flask/FastAPI based on the `server.py` and `templates` structure) to serve HTML templates and static assets (CSS, JS).

## File and Subfolder List

* `__init__.py`: Package initialization file. (Role: **Configuration Analyst**)
* `server.py`: The main server application script that hosts the backend logic and routing for the dashboard. (Role: **Runtime Investigator, API Analyst**)
* `static/`: Subfolder containing static web assets. (Role: **Data Model Analyst**)
  * `app.js`: Client-side JavaScript for interactive UI elements. (Role: **Runtime Investigator**)
  * `style.css`: Stylesheet for the web dashboard. (Role: **Data Model Analyst**)
* `templates/`: Subfolder containing HTML templates for the dashboard views. (Role: **Data Model Analyst**)
  * `ai_os.html`: Template for AI OS view. (Role: **Data Model Analyst**)
  * `base.html`: Base layout template. (Role: **Data Model Analyst**)
  * `clicker.html`: Template for clicker functionality. (Role: **Data Model Analyst**)
  * `converter.html`: Template for converter tool. (Role: **Data Model Analyst**)
  * `goals.html`: Template for user goals. (Role: **Data Model Analyst**)
  * `health.html`: Template for system or user health monitoring. (Role: **Data Model Analyst**)
  * `index.html`: Main dashboard entry point. (Role: **Data Model Analyst**)
  * `login.html`: Authentication view. (Role: **Data Model Analyst**)
  * `memory.html`: Template to display or edit AI memory. (Role: **Data Model Analyst, Prompt Recovery Specialist**)
  * `search.html`: Template for search functionality. (Role: **Data Model Analyst**)

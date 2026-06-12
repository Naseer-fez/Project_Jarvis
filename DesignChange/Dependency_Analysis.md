# Dependency Analysis

## Package: audit
- **Imports**: __future__, audit_logger, re
- **Imported By**: None
- **External Dependencies**: (Evaluated via standard library exclusion heuristic)
- **Coupling Score**: 3
- **Centrality Score**: 0

## Package: core
- **Imports**: PIL, __future__, abc, aiohttp, aiosqlite, argparse, ast, asyncio, atexit, base64, bcrypt, bs4, chromadb, collections, concurrent, confidence, configparser, contextlib, contextvars, copy, core, csv, ctypes, cv2, dataclasses, datetime, device_registry, duckduckgo_search, edge_tts, enum, faster_whisper, fnmatch, fpdf, functools, hashlib, health, hmac, html, httpx, importlib, inspect, io, json, logging, markdown, math, numpy, os, pandas, parsedatetime, pathlib, platform, plyer, psutil, pvporcupine, pvrecorder, pyautogui, pygetwindow, pypdf, pyperclip, pytesseract, pyttsx3, queue, re, requests, secrets, sentence_transformers, serial, serial_controller, shlex, shutil, sounddevice, speech_recognition, sqlite3, streamlit, struct, subprocess, sys, tempfile, textwrap, threading, time, torch, traceback, typing, urllib, uuid, yaml
- **Imported By**: core, dashboard, integrations, root, tests
- **External Dependencies**: (Evaluated via standard library exclusion heuristic)
- **Coupling Score**: 93
- **Centrality Score**: 5

## Package: dashboard
- **Imports**: __future__, asyncio, contextlib, core, dataclasses, datetime, fastapi, hmac, logging, os, pathlib, pydantic, shutil, sqlite3, starlette, sys, tempfile, threading, time, typing, uuid, uvicorn
- **Imported By**: tests
- **External Dependencies**: (Evaluated via standard library exclusion heuristic)
- **Coupling Score**: 23
- **Centrality Score**: 1

## Package: integrations
- **Imports**: __future__, abc, aiohttp, asyncio, base64, core, datetime, dateutil, email, github, icalendar, imaplib, importlib, inspect, integrations, json, logging, os, pathlib, pyautogui, smtplib, telegram, time, twilio, typing, urllib
- **Imported By**: integrations, tests
- **External Dependencies**: (Evaluated via standard library exclusion heuristic)
- **Coupling Score**: 28
- **Centrality Score**: 2

## Package: root
- **Imports**: __future__, aiosqlite, argparse, ast, asyncio, collections, core, json, logging, os, pathlib, re, requests, shutil, signal, subprocess, sys, threading, time, traceback
- **Imported By**: None
- **External Dependencies**: (Evaluated via standard library exclusion heuristic)
- **Coupling Score**: 20
- **Centrality Score**: 0

## Package: tests
- **Imports**: __future__, aiohttp, asyncio, configparser, contextlib, core, dashboard, fastapi, gc, importlib, inspect, integrations, json, logging, os, pathlib, pytest, sqlite3, sys, tempfile, threading, time, types, typing, unittest, uuid
- **Imported By**: None
- **External Dependencies**: (Evaluated via standard library exclusion heuristic)
- **Coupling Score**: 26
- **Centrality Score**: 0


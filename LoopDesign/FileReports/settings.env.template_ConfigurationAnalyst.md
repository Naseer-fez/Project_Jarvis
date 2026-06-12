# File Report: settings.env.template
**Role:** Configuration Analyst

## File Overview
This file serves as a template for the `settings.env` file. It outlines all available environment variables that a user can define to enable various external integrations.

## Assumptions & Contracts
- **Format:** `.env` style key-value pairs.
- **Implicit Environment Assumptions:** Same as `settings.env` (Gmail SMTP/IMAP defaults, Home Assistant local mDNS default, Twilio sandbox default).

## Secrets & Env Vars
Contains the exact same structure as `settings.env`, but with two additional variables appended at the end:
- `HF_TOKEN` (Hugging Face token)
- `TAVILY_API_KEY` (Tavily search API key, which also appears blank in `jarvis.ini`)

## Extracted Prompts
- None.

## Configuration Variables
Template values provided for structural understanding. No concrete secrets are populated other than the system defaults (URLs, Ports).

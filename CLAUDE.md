# Recall Echo Monitor
Compares US recalls (NHTSA) with India recalls (SIAM) for carmakers that sell the same models in both markets.

# Rules
- Python 3.11, pandas, duckdb. SQL queries live in sql/, one file per KPI
- Never edit files in data/raw. Write outputs to data/clean
- Every function in src/ gets a pytest test in tests/. Run `pytest -q` after each change
- IMPORTANT: never invent data or fill gaps with guesses. If a value is missing, keep it null and log it
- Read the API key from .env. Never print or commit it
- Explain each new module to me in 5 plain lines after writing it

# Domain notes
- A missing India recall is a transparency signal, not proof of wrongdoing
- Field meanings are in data/raw/RCL.txt

# Commands (Windows)
- Run modules with `.venv/Scripts/python.exe -m src.<module>`
- Tests: `.venv/Scripts/python.exe -m pytest -q`
- Database: data/clean/recalls.duckdb

# Matching rules
- Unit of comparison: one NHTSA campaign (CAMPNO) vs one SIAM row
- Brands are compared through the `make` column, never manufacturer_name
- Mercedes SIAM models use platform codes: data/reference/mercedes_platform_codes.csv
- LLM calls go through src/llm.py only, batched, never one call per row
- Every rate shown anywhere must show its n. Suppress rates where n < 10
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def check_file_exists(label: str, path: Path) -> bool:
    if path.exists():
        print(f"[OK] {label}: {path}")
        return True

    print(f"[MISSING] {label}: {path}")
    return False


def check_env_var(name: str) -> bool:
    value = os.getenv(name)

    if value:
        print(f"[OK] {name}: set")
        return True

    print(f"[MISSING] {name}: not set")
    return False


def check_google_endpoint(url: str, timeout_seconds: int = 20) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout_seconds)
        print(f"[OK] Google endpoint reachable: {url}")
        return True
    except urllib.error.HTTPError as error:
        # Root Google API endpoints often return 404.
        # 404 still means HTTPS/network connection worked.
        if error.code == 404:
            print(f"[OK] Google endpoint reachable: {url} returned 404")
            return True

        print(f"[FAILED] Google endpoint HTTP error: {url} -> {error}")
        return False
    except Exception as error:
        print(f"[FAILED] Google endpoint not reachable: {url} -> {type(error).__name__}: {error}")
        return False


def run_live_sheets_check() -> bool:
    try:
        from core.tools.google_sheets_tools import get_sheet_headers

        headers = get_sheet_headers("Clients")
        print(f"[OK] Google Sheets live check: Clients headers = {headers}")
        return True
    except Exception as error:
        print(f"[FAILED] Google Sheets live check: {type(error).__name__}: {error}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also call Google Sheets API and read the Clients headers.",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    credentials_dir = PROJECT_ROOT / "credentials"
    credentials_file = credentials_dir / "google_calendar_credentials.json"
    token_file = credentials_dir / "google_oauth_token.json"

    print("Sira Google integration health check")
    print("-" * 40)

    checks = []

    checks.append(check_file_exists("Credentials file", credentials_file))
    checks.append(check_file_exists("OAuth token file", token_file))
    checks.append(check_env_var("GOOGLE_SHEETS_SPREADSHEET_ID"))
    checks.append(check_google_endpoint("https://www.googleapis.com"))
    checks.append(check_google_endpoint("https://sheets.googleapis.com"))

    if args.live:
        checks.append(run_live_sheets_check())
    else:
        print("[SKIP] Google Sheets live check. Run with --live to enable it.")

    print("-" * 40)

    if all(checks):
        print("[OK] Google integration looks healthy.")
        return 0

    print("[FAILED] One or more checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
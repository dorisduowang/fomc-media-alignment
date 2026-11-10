"""
MACS 30113

Yuzhao (Jerry) Jin; CNETID: yuzhaojin1
Doris (Duo) Wang; CNETID: dorisduowang

This file is the set up file for all project-wide environment variables.
All local script imports from here instead of reading os.environ directly.

Reads in priority order:
  1. Explicit CLI argument passed to the script
  2. .env file in the project root
  3. Windows / Mac / Linux system env vars
  4. Empty string / sensible default

Usage in any script:
    from config import BUCKET, NYT_KEYS, GDN_KEY, AWS_REGION
"""

import os
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def _require(key: str) -> str:
    """Read a required variable; raise a clear error if missing."""
    val = os.environ.get(key, "").strip()
    if not val:
        raise EnvironmentError(
            f"\n[config] Required variable '{key}' is not set.\n"
            f"  Set it in your .env file (see .env.example) or as a\n"
            f"  system environment variable before running this script.\n"
        )
    return val

def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

# Project-wide variables — imported by all local scripts

BUCKET      = _optional("BUCKET")           # S3 lake bucket name
AWS_REGION  = _optional("AWS_DEFAULT_REGION", "us-east-1")
GDN_KEY     = _optional("GDN_KEY")          # Guardian Open Platform API key

# NYT keys are stored as a comma-separated list to support multiple keys.
_nyt_raw    = _optional("NYT_KEYS")
NYT_KEYS    = [k.strip() for k in _nyt_raw.split(",") if k.strip()]


def require_bucket() -> str:
    """Call this in any script that must have BUCKET set."""
    if not BUCKET:
        raise EnvironmentError(
            "[config] BUCKET is not set. Add it to your .env file:\n"
            "  BUCKET=your-fomc-bucket\n"
        )
    return BUCKET

def require_nyt_keys() -> list:
    """Call this in the NYT ingestion script."""
    if not NYT_KEYS:
        raise EnvironmentError(
            "[config] NYT_KEYS is not set. Add it to your .env file:\n"
            "  NYT_KEYS=your_key_1,your_key_2,your_key_3\n"
        )
    return NYT_KEYS

def require_gdn_key() -> str:
    """Call this in the Guardian ingestion script."""
    if not GDN_KEY:
        raise EnvironmentError(
            "[config] GDN_KEY is not set. Add it to your .env file:\n"
            "  GDN_KEY=your_guardian_api_key\n"
        )
    return GDN_KEY

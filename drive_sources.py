"""Read spreadsheets from a private Google Drive folder via a service account.

Supports Google Sheets (any tab), .xlsx and .csv files, including one or two
levels of sub-folders. Google client libraries are imported lazily so the pure
helpers below can be tested without them.
"""
import io
import re

import pandas as pd

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]
FOLDER = "application/vnd.google-apps.folder"
GSHEET = "application/vnd.google-apps.spreadsheet"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CSV = "text/csv"
SUPPORTED = {GSHEET, XLSX, CSV}
MAX_DEPTH = 2  # folder levels below the one you configure


def build_services(info: dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive, sheets


def folder_id_from(value: str) -> str:
    """Accepts a full folder URL or a bare folder ID."""
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", value) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", value)
    return m.group(1) if m else value.strip()


def list_files(drive, folder_id: str, prefix: str = "", depth: int = 0) -> list[dict]:
    """Spreadsheet files in a folder (and sub-folders), each with a display 'path'."""
    q = f"'{folder_id}' in parents and trashed = false"
    out, token = [], None
    while True:
        resp = drive.files().list(
            q=q, pageSize=200, pageToken=token, orderBy="name",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == FOLDER and depth < MAX_DEPTH:
                out += list_files(drive, f["id"], f"{prefix}{f['name']}/", depth + 1)
            elif f["mimeType"] in SUPPORTED:
                out.append({**f, "path": f"{prefix}{f['name']}"})
        token = resp.get("nextPageToken")
        if not token:
            return out


def list_tabs(sheets, file_id: str) -> list[str]:
    meta = sheets.spreadsheets().get(
        spreadsheetId=file_id, fields="sheets.properties.title").execute()
    return [s["properties"]["title"] for s in meta.get("sheets", [])]


def peek_columns(sheets, file_id: str) -> list[str]:
    """Column headers of the first tab only (one cheap API call; range '1:1' = first sheet, row 1)."""
    resp = sheets.spreadsheets().values().get(spreadsheetId=file_id, range="1:1").execute()
    rows = resp.get("values") or [[]]
    return [str(c).strip() for c in rows[0] if str(c).strip()]


def rows_to_frame(rows: list[list]) -> pd.DataFrame:
    """Sheets API returns ragged rows; first row is the header."""
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    header = [str(h).strip() for h in rows[0]] + [""] * (width - len(rows[0]))
    names, seen = [], {}
    for i, h in enumerate(header):
        h = h or f"Column {i + 1}"
        n = seen.get(h, 0)
        seen[h] = n + 1
        names.append(h if n == 0 else f"{h}_{n + 1}")
    body = [list(r) + [None] * (width - len(r)) for r in rows[1:]]
    return pd.DataFrame(body, columns=names).replace("", pd.NA)


def read_gsheet(sheets, file_id: str, tab: str | None = None) -> pd.DataFrame:
    if tab is None:
        tab = list_tabs(sheets, file_id)[0]
    rng = "'" + tab.replace("'", "''") + "'"
    resp = sheets.spreadsheets().values().get(spreadsheetId=file_id, range=rng).execute()
    return rows_to_frame(resp.get("values", []))


def read_upload(drive, file_id: str, mime: str) -> pd.DataFrame:
    """Download an .xlsx or .csv that was dropped into the folder (xlsx: first sheet)."""
    from googleapiclient.http import MediaIoBaseDownload

    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, drive.files().get_media(fileId=file_id, supportsAllDrives=True))
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return pd.read_excel(buf) if mime == XLSX else pd.read_csv(buf)


def read_file(drive, sheets, f: dict, tab: str | None = None) -> pd.DataFrame:
    if f["mimeType"] == GSHEET:
        return read_gsheet(sheets, f["id"], tab)
    return read_upload(drive, f["id"], f["mimeType"])


VERSION = 6  # bump together with app.py REQUIRED_VERSION so stale copies are caught at startup

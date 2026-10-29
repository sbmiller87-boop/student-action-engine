"""Describe how current the data behind an answer is. No Streamlit dependency."""
from datetime import datetime, timezone

STALE_DAYS = 14  # warn if the oldest file behind an answer hasn't been edited for this long


def parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def ago(dt: datetime, now: datetime) -> str:
    s = max(0, int((now - dt).total_seconds()))
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        h = s // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = s // 86400
    if d < 14:
        return f"{d} day{'s' if d != 1 else ''} ago"
    if d < 60:
        return f"{d // 7} weeks ago"
    return f"{d // 30} months ago"


def when(dt: datetime, tz, now: datetime | None = None) -> str:
    local = dt.astimezone(tz)
    h = local.hour % 12 or 12
    ampm = "am" if local.hour < 12 else "pm"
    year = f" {local.year}" if now and local.year != now.astimezone(tz).year else ""
    return f"{local:%a} {local.day} {local:%b}{year}, {h}:{local.minute:02d} {ampm}"


def summarize(metas: list[dict], tz, now: datetime | None = None) -> dict:
    """metas: [{label, modified (ISO or None), fetched_at (ISO)}].
    Returns {headline, stale, rows}."""
    now = now or datetime.now(timezone.utc)
    rows, edits, fetches = [], [], []
    for m in metas:
        mod, fet = parse(m.get("modified")), parse(m.get("fetched_at"))
        if mod:
            edits.append(mod)
        if fet:
            fetches.append(fet)
        rows.append({
            "file": m["label"],
            "last edited": f"{when(mod, tz, now)} ({ago(mod, now)})" if mod else "not available (link sheet)",
            "synced from Google": f"{when(fet, tz, now)} ({ago(fet, now)})" if fet else "",
        })

    synced = f"synced {ago(min(fetches), now)}" if fetches else ""
    stale = bool(edits) and (now - min(edits)).days >= STALE_DAYS
    unknown = len(metas) - len(edits)

    if len(metas) == 1:
        m = metas[0]
        if edits:
            head = f"📅 {m['label']}: last edited {when(edits[0], tz, now)} ({ago(edits[0], now)})"
        else:
            head = f"📅 {m['label']}: last-edited time isn't available for link sheets"
    else:
        head = f"📅 {len(metas)} files"
        if edits:
            newest, oldest = max(edits), min(edits)
            head += (f": most recent edit {when(newest, tz, now)} ({ago(newest, now)}), "
                     f"oldest {when(oldest, tz, now)} ({ago(oldest, now)})")
        if unknown:
            head += f"{' ·' if edits else ':'} {unknown} link sheet(s) with no edit time"
    if synced:
        head += f" · {synced}"
    if stale:
        head += f" · ⚠️ some of this data hasn't been edited for {STALE_DAYS}+ days"
    return {"headline": head, "stale": stale, "rows": rows}


VERSION = 6  # bump together with app.py REQUIRED_VERSION so stale copies are caught at startup

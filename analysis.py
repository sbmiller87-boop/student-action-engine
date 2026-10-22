"""Pure pandas/plotly analysis helpers (no Streamlit) used by app.py."""
import numpy as np
import pandas as pd
import plotly.express as px

MIN_N = 10          # minimum marks before we make comparisons / raise flags
EFFECT_FLAG = 0.5   # difference in peer-SDs worth flagging
HEAP_FLAG = 40      # % of marks identical worth flagging


# ---------------------------------------------------------------- helpers
def numeric_view(s: pd.Series) -> pd.Series:
    """Numeric version of a column; text like 'Abs' or blanks become NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype(str).str.replace(r"[%,$]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def profile(df: pd.DataFrame) -> list[dict]:
    """Compact schema description sent to the AI instead of the whole sheet."""
    cols = []
    for c in df.columns:
        s = df[c]
        info = {"name": c, "dtype": str(s.dtype), "missing": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            info.update(min=float(s.min()), max=float(s.max()), mean=round(float(s.mean()), 2))
        else:
            u = s.dropna().unique()
            info["unique"] = int(len(u))
            info["examples"] = [str(x) for x in u[:8]]
            nonnull = s.notna().sum()
            if nonnull:
                share = numeric_view(s).notna().sum() / nonnull
                if share > 0.3:
                    info["numeric_share"] = round(float(share), 2)  # mixed number/text column
        cols.append(info)
    return cols


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    out = df
    valid = set(df.columns)
    for f in filters or []:
        c, o, v = f.get("column"), f.get("op"), f.get("value")
        if c not in valid:
            raise ValueError(f"Unknown column '{c}'")
        s = out[c]
        if o in (">", ">=", "<", "<="):
            s = numeric_view(s)
            v = float(v)
        is_text = not pd.api.types.is_numeric_dtype(s)
        if o == "in" and not isinstance(v, list):
            v = [v]

        def eq():
            if is_text:
                return s.astype(str).str.strip().str.lower() == str(v).strip().lower()
            try:
                return s == float(v)
            except (TypeError, ValueError):
                return pd.Series(False, index=s.index)

        masks = {
            "==": eq,
            "!=": lambda: ~eq(),
            ">": lambda: s > v, ">=": lambda: s >= v, "<": lambda: s < v, "<=": lambda: s <= v,
            "contains": lambda: s.astype(str).str.contains(str(v), case=False, na=False),
            "in": lambda: s.astype(str).str.strip().str.lower().isin([str(x).strip().lower() for x in v]),
            "isnull": lambda: s.isna(), "notnull": lambda: s.notna(),
        }
        if o not in masks:
            raise ValueError(f"Unsupported operator '{o}'")
        out = out[masks[o]()]
    return out


def describe(vals: pd.Series, total: int) -> dict:
    v = vals.dropna()
    d = {"n": int(total), "numeric_n": int(len(v)),
         "non_numeric_%": round(100 * (1 - len(v) / total), 1) if total else 0.0}
    if len(v):
        top = v.round(2).value_counts()
        d.update(mean=round(float(v.mean()), 2), median=round(float(v.median()), 2),
                 sd=round(float(v.std()), 2) if len(v) > 1 else 0.0,
                 min=round(float(v.min()), 2), max=round(float(v.max()), 2),
                 top_value=float(top.index[0]), **{"top_value_%": round(100 * top.iloc[0] / len(v), 1)})
    else:
        d.update(mean=np.nan, median=np.nan, sd=np.nan, min=np.nan, max=np.nan,
                 top_value=np.nan, **{"top_value_%": 0.0})
    return d


def flag_list(name: str, fs: dict, rs: dict | None, peers: str) -> list[str]:
    out = []
    if fs["numeric_n"] < MIN_N:
        return out
    if fs["top_value_%"] >= HEAP_FLAG:
        out.append(f"{name}: {fs['top_value_%']:.0f}% of marks are the identical value ({fs['top_value']:g}).")
    if rs and rs["numeric_n"] >= MIN_N and rs["sd"] and rs["sd"] > 0:
        eff = (fs["mean"] - rs["mean"]) / rs["sd"]
        if abs(eff) >= EFFECT_FLAG:
            out.append(f"{name}: average {fs['mean']:g} is {'higher' if eff > 0 else 'lower'} than "
                       f"{peers} ({rs['mean']:g}), about {abs(eff):.1f} peer standard deviations.")
        ratio = fs["sd"] / rs["sd"]
        if ratio < 0.5 or ratio > 1.6:
            out.append(f"{name}: marks are unusually {'tightly clustered' if ratio < 0.5 else 'spread out'} "
                       f"(SD {fs['sd']:g} vs {rs['sd']:g} for {peers}).")
        if fs["non_numeric_%"] - rs["non_numeric_%"] >= 10:
            out.append(f"{name}: {fs['non_numeric_%']:.0f}% blank or non-numeric entries vs "
                       f"{rs['non_numeric_%']:.0f}% for {peers}.")
    return out


# ---------------------------------------------------------------- main entry
def run_plan(df: pd.DataFrame, plan: dict) -> dict:
    """Returns {result, figs, stats, levels, flags, notes, summary}."""
    if plan.get("intent") == "distribution":
        if plan.get("measure") not in df.columns:
            raise ValueError(f"Unknown measure column '{plan.get('measure')}'")
        return _distribution(df, plan)
    return _table_or_aggregate(df, plan)


def _table_or_aggregate(df, plan):
    valid = set(df.columns)
    out = apply_filters(apply_filters(df, plan.get("filters")), plan.get("focus"))
    figs, summary = [], None
    agg = plan.get("aggregate")
    if agg:
        if agg.get("column") not in valid:
            raise ValueError(f"Unknown column '{agg.get('column')}'")
        keys = [g for g in plan.get("group_by") or [] if g in valid]
        func = agg.get("func", "mean")
        if func not in {"mean", "median", "sum", "count", "min", "max"}:
            raise ValueError(f"Unsupported aggregation '{func}'")
        col = agg["column"]
        src = out[col] if func == "count" else numeric_view(out[col])
        tmp = out[keys].copy() if keys else pd.DataFrame(index=out.index)
        tmp[col] = src
        out = (tmp.groupby(keys)[col].agg(func).reset_index() if keys
               else pd.DataFrame({col: [tmp[col].agg(func)]}))
        if len(keys) == 1 and 2 <= len(out) <= 60:
            figs.append(px.bar(out, x=keys[0], y=col, title=f"{func.title()} of {col} by {keys[0]}"))
    else:
        cols = [c for c in plan.get("select_columns") or [] if c in valid]
        if cols:
            out = out[cols]
    sb = plan.get("sort_by")
    if sb in out.columns:
        out = out.sort_values(sb, ascending=bool(plan.get("ascending", True)))
    if plan.get("limit"):
        out = out.head(int(plan["limit"]))
    if agg and len(out) <= 40:
        summary = out.to_csv(index=False)
    return dict(result=out, figs=figs, stats=None, levels=None, flags=[], notes=[], summary=summary)


def _distribution(df, plan):
    measure = plan["measure"]
    by = plan.get("by") if plan.get("by") in df.columns else None
    label = str(plan.get("focus_label") or "Selected group")
    peers = "peers"
    ctx = apply_filters(df, plan.get("filters"))
    has_focus = bool(plan.get("focus"))
    focus = apply_filters(ctx, plan.get("focus")) if has_focus else ctx
    if focus.empty:
        raise ValueError("No rows matched the group you asked about. Check the name or code spelling.")
    rest = ctx.drop(focus.index) if has_focus else ctx.iloc[0:0]

    notes, like_for_like = [], False
    if has_focus and by:
        shared = rest[rest[by].isin(focus[by].dropna().unique())]
        if len(shared) >= MIN_N:
            rest, like_for_like = shared, True
            notes.append(f"Peers are other rows with the same '{by}' values, so comparisons are like-for-like.")
        else:
            notes.append(f"Few or no peers share the same '{by}' values, so overall charts compare against "
                         f"everyone else in the sheet, which may not be like-for-like.")
    elif has_focus:
        notes.append("Peers are all other rows in the sheet.")

    nn = ctx[measure].notna().sum()
    numeric = nn > 0 and numeric_view(ctx[measure]).notna().sum() / nn >= 0.8
    figs, flags = [], []

    if numeric:
        fv, rv = numeric_view(focus[measure]), numeric_view(rest[measure])
        fs, rs = describe(fv, len(focus)), (describe(rv, len(rest)) if len(rest) else None)
        stats = pd.DataFrame([{"group": label, **fs}] + ([{"group": "Peers", **rs}] if rs else []))
        flags += flag_list("Overall", fs, rs, peers)

        def frame(src, vals, group):
            d = pd.DataFrame({"value": vals, "group": group})
            if by:
                d[by] = src[by].astype(str).str.slice(0, 30)
            return d.dropna(subset=["value"])

        comb = pd.concat([frame(focus, fv, label)] + ([frame(rest, rv, "Peers")] if len(rest) else []))
        figs.append(px.histogram(comb, x="value", color="group", barmode="overlay", histnorm="percent",
                                 opacity=0.6, nbins=30,
                                 title=f"Distribution of {measure}: {label} vs {peers}",
                                 labels={"value": measure, "percent": "% of marks"}))

        levels = None
        if by:
            rows = []
            for lvl in focus[by].value_counts().index[:40]:
                fsub = focus[focus[by] == lvl]
                f = numeric_view(fsub[measure])
                rsub = rest[rest[by] == lvl] if like_for_like else rest.iloc[0:0]
                r = numeric_view(rsub[measure])
                fd = describe(f, len(fsub))
                rd = describe(r, len(rsub)) if len(rsub) else None
                eff = np.nan
                if rd and rd["numeric_n"] >= MIN_N and fd["numeric_n"] >= MIN_N and rd["sd"] > 0:
                    eff = round((fd["mean"] - rd["mean"]) / rd["sd"], 2)
                rows.append({by: lvl, "n": fd["n"], "mean": fd["mean"], "sd": fd["sd"],
                             "top_value_%": fd["top_value_%"],
                             "peer_n": len(rsub), "peer_mean": rd["mean"] if rd else np.nan,
                             "peer_sd": rd["sd"] if rd else np.nan, "effect_sd": eff})
                flags += flag_list(str(lvl), fd, rd, peers)
            levels = pd.DataFrame(rows)

            top_levels = focus[by].astype(str).str.slice(0, 30).value_counts().index[:12]
            sub = comb[comb[by].isin(top_levels)]
            if len(sub):
                figs.append(px.box(sub, x=by, y="value", color="group",
                                   title=f"{measure} by {by} (12 most-marked)",
                                   labels={"value": measure}))
            comp = levels.dropna(subset=["effect_sd"])
            if len(comp):
                comp = comp.reindex(comp["effect_sd"].abs().sort_values(ascending=False).index).head(25)
                comp = comp.sort_values("effect_sd")
                fig = px.bar(comp, x="effect_sd", y=by, orientation="h",
                             title=f"Average vs {peers}, in peer standard deviations (±{EFFECT_FLAG} = worth a look)",
                             labels={"effect_sd": "difference (SD)"})
                fig.add_vline(x=EFFECT_FLAG, line_dash="dot")
                fig.add_vline(x=-EFFECT_FLAG, line_dash="dot")
                figs.append(fig)
        result = levels if levels is not None else stats
    else:
        # categorical marks (e.g. grades)
        fc = focus[measure].astype(str).value_counts(normalize=True) * 100
        rc = rest[measure].astype(str).value_counts(normalize=True) * 100 if len(rest) else pd.Series(dtype=float)
        cats = fc.add(rc, fill_value=0).sort_values(ascending=False).index[:15]
        tbl = pd.DataFrame({measure: cats, label: [round(fc.get(c, 0), 1) for c in cats]})
        if len(rest):
            tbl["Peers"] = [round(rc.get(c, 0), 1) for c in cats]
            for _, r in tbl.iterrows():
                if len(focus) >= 20 and abs(r[label] - r["Peers"]) >= 10:
                    flags.append(f"{r[measure]}: {r[label]:.0f}% of {label}'s results vs {r['Peers']:.0f}% for {peers}.")
        long = tbl.melt(id_vars=measure, var_name="group", value_name="percent")
        figs.append(px.bar(long, x=measure, y="percent", color="group", barmode="group",
                           title=f"Share of each {measure} value: {label} vs {peers}"))
        stats, levels, result = tbl, None, tbl

    summary = f"Focus: {label} ({len(focus)} rows); peers: {len(rest)} rows. Measure: {measure}"
    summary += f"{' by ' + by if by else ''}. {'Numeric' if numeric else 'Categorical'} marks.\n"
    summary += "Overall stats:\n" + stats.to_csv(index=False)
    if levels is not None and len(levels):
        top = levels.reindex(levels["effect_sd"].abs().sort_values(ascending=False, na_position="last").index).head(15)
        summary += "\nPer-level comparison (largest differences first):\n" + top.to_csv(index=False)
    summary += "\nAutomatic flags:\n" + ("\n".join(flags[:20]) if flags else "none")
    summary += "\nNotes:\n" + "\n".join(notes)
    return dict(result=result, figs=figs, stats=stats, levels=levels, flags=flags, notes=notes, summary=summary)

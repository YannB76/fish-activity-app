# -*- coding: utf-8 -*-
"""
App Streamlit – Score d’activité (Bar) – Le Havre – 72 h

SOURCES:
- Météo: Open-Meteo Forecast (pression, vent, nébulosité, température air, weathercode)
- Mer:   Open-Meteo Marine (hauteur, direction, période) – fallback hauteur seule
- Astro: Open-Meteo Astronomy si dispo; sinon fallback local 'astral'
- Marées: WorldTides (clé) – sinon bonus marée=0 (UI l’indique) + pas de prochaine PM
- SST (temp. de l’eau): priorité Copernicus Marine → API locale FastAPI → cabaigne.net (scrape) → Open-Meteo water_temperature → NaN

UI:
- Bandeau “Provenance des données”
- Récap météo/mer/astro/marées (incl. prochaine pleine mer)
- Top créneaux & 3 graphiques (un/jour) – **Bar uniquement**
"""

import os
import re
import math
import time
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from requests.exceptions import HTTPError, RequestException
from bs4 import BeautifulSoup  # pour le scrape cabaigne

# -----------------------------
# CONFIG GÉNÉRALE
# -----------------------------
st.set_page_config(page_title="🎣 Activité Bar – Le Havre (72 h)", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Paris")
LAT, LON = 49.494, 0.107       # Le Havre
HOURS = 72

def get_secret(name: str, default=None):
    """Essaie st.secrets[name], sinon variable d'env, sinon défaut."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🎣 Score d’activité – Bar (72 h) – Le Havre")
st.caption("Sources: Open-Meteo (météo/mer/astro), Copernicus Marine (SST prioritaire), API locale SST/cabaigne/Open-Meteo (fallback), WorldTides (marées).")

with st.expander("⚙️ Options"):
    min_score = st.slider("Seuil de créneau conseillé (≥)", 40, 90, 60, 5)
    show_details = st.checkbox("Afficher le tableau des données horaires", value=False)

# -----------------------------
# SCORING HELPERS (Bar)
# -----------------------------
def is_onshore(wind_dir_deg: float) -> bool:
    if wind_dir_deg is None or np.isnan(wind_dir_deg): return False
    wd = wind_dir_deg % 360
    return (270 <= wd <= 360) or (0 <= wd <= 60)

def linear_window(x, x1, x2):
    if x is None or np.isnan(x): return 0.0
    if x1 > x2: x1, x2 = x2, x1
    if x < x1 or x > x2: return 0.0
    mid = 0.5*(x1+x2); span = 0.5*(x2-x1)
    if span <= 0: return 1.0
    return max(0.0, 1.0 - abs(x-mid)/span)

def temp_bonus_bar(temp_c: float) -> float:
    if temp_c is None or np.isnan(temp_c): return 0.0
    # Bar: zone “idéale” 12–18 °C
    w = linear_window(temp_c, 12, 18)
    return 10.0 * w

def wave_bonus_bar(hs_m: float) -> float:
    if hs_m is None or np.isnan(hs_m): return 0.0
    if 0.4 <= hs_m <= 1.2: return 10.0
    if hs_m < 0.2: return -5.0
    if hs_m > 2.0: return -15.0
    return 0.0

def wind_bonus_bar(wind_speed_ms: float, wind_dir_deg: float) -> float:
    if any(x is None or np.isnan(x) for x in [wind_speed_ms, wind_dir_deg]): return 0.0
    b = 0.0
    if is_onshore(wind_dir_deg):
        if 3.0 <= wind_speed_ms <= 8.0: b += 15.0
        elif wind_speed_ms > 12.0: b -= 10.0
    else:
        b -= 10.0
    return b

def pressure_bonus(p_now: float, p_trend_6h: float) -> float:
    if p_now is None or np.isnan(p_now): return 0.0
    trend = p_trend_6h if (p_trend_6h is not None and not np.isnan(p_trend_6h)) else 0.0
    if trend <= -6.0: return 20.0
    if -2.0 < trend < 2.0: return -5.0
    if trend >= 4.0: return -10.0
    return 5.0 if trend < -2.0 else 0.0

def tide_bonus(tide_state: str | None, minutes_to_high: float | None) -> float:
    if not tide_state and minutes_to_high is None: return 0.0
    b = 0.0
    ts = (tide_state or "").lower()
    if any(k in ts for k in ["rising","mont","flood"]): b += 20.0
    elif any(k in ts for k in ["falling","desc","ebb"]): b += 5.0
    elif any(k in ts for k in ["slack","etale"]): b -= 15.0
    if minutes_to_high is not None and not np.isnan(minutes_to_high):
        if abs(minutes_to_high) <= 120: b += 10.0
    return b

def cloud_bonus_bar(cloud_pct: float, is_daytime: bool) -> float:
    if cloud_pct is None or np.isnan(cloud_pct): return 0.0
    # Le bar supporte un ciel chargé en journée (lumière diffuse)
    return 5.0 if (is_daytime and cloud_pct >= 50) else 0.0

def tod_bonus_bar(ts_local: datetime, sunrise: datetime | None, sunset: datetime | None) -> float:
    bonus = 0.0
    if sunrise and sunset:
        def hdiff(a,b): return abs((a-b).total_seconds())/3600
        if hdiff(ts_local, sunrise)<=1 or hdiff(ts_local, sunset)<=1: bonus += 15.0
        # Bonus nuit
        if ts_local<sunrise or ts_local>sunset: bonus += 10.0
    else:
        h = ts_local.hour + ts_local.minute/60
        if abs(h-7.0)<=1.0 or abs(h-21.0)<=1.0: bonus += 15.0
        if h>=23 or h<=5: bonus += 10.0
    return bonus

def moon_bonus(moon_phase_0_new_1_full: float) -> float:
    """Bonus léger en nouvelle ou pleine lune (≈ ±10% du cycle)."""
    try:
        val = float(moon_phase_0_new_1_full)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(val):
        return 0.0
    return 5.0 if (val <= 0.1 or val >= 0.9) else 0.0

# -----------------------------
# OUTILS D’AFFICHAGE METEO
# -----------------------------
WEATHERCODE_MAP = {
    0: "Ciel clair",
    1: "Plutôt clair", 2: "Partiellement nuageux", 3: "Couvert",
    45: "Brouillard", 48: "Brouillard givr.",
    51: "Bruine légère", 53: "Bruine", 55: "Bruine forte",
    61: "Pluie légère", 63: "Pluie", 65: "Pluie forte",
    66: "Pluie verglaçante légère", 67: "Pluie verglaçante",
    71: "Neige légère", 73: "Neige", 75: "Neige forte",
    77: "Granules de neige",
    80: "Averses faibles", 81: "Averses", 82: "Averses fortes",
    85: "Averses neige faibles", 86: "Averses neige fortes",
    95: "Orages", 96: "Orage grésil", 97: "Orage grésil fort",
}

def code_to_desc(wcode):
    try:
        code = int(wcode)
        return WEATHERCODE_MAP.get(code, f"Code météo {code}")
    except Exception:
        return "—"

# -----------------------------
# OPEN-METEO – Forecast
# -----------------------------
def fetch_openmeteo_forecast(lat, lon, hours=HOURS, tz="Europe/Paris"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,pressure_msl,windspeed_10m,winddirection_10m,cloudcover,weathercode",
        "timezone": tz, "forecast_days": 4
    }
    r = requests.get(url, params=params, timeout=20); r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(h["time"]),
        "air_temp_c": h["temperature_2m"],
        "pressure_hpa": h["pressure_msl"],
        "wind_speed_ms": np.array(h["windspeed_10m"])/3.6,
        "wind_dir_deg": h["winddirection_10m"],
        "cloud_cover_pct": h["cloudcover"],
        "weathercode": h["weathercode"],
    })
    df["datetime"] = df["datetime"].dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    now = datetime.now(LOCAL_TZ)
    return df[(df["datetime"] >= now - timedelta(hours=1)) & (df["datetime"] <= now + timedelta(hours=hours))].reset_index(drop=True)

# -----------------------------
# OPEN-METEO – Marine (robuste)
# -----------------------------
def fetch_openmeteo_marine(lat, lon, hours=HOURS, tz="Europe/Paris"):
    base_url = "https://marine-api.open-meteo.com/v1/marine"
    now = datetime.now(LOCAL_TZ)

    def _time_filter(df):
        df["datetime"] = df["datetime"].dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
        return df[(df["datetime"] >= now - timedelta(hours=1)) & (df["datetime"] <= now + timedelta(hours=hours))].reset_index(drop=True)

    # 1) complet
    try:
        params = {"latitude": lat, "longitude": lon, "hourly": "wave_height,wave_direction,wave_period", "timezone": tz, "forecast_days": 4}
        r = requests.get(base_url, params=params, timeout=20); r.raise_for_status()
        j = r.json(); h = j.get("hourly", {})
        df = pd.DataFrame({
            "datetime": pd.to_datetime(h.get("time", [])),
            "wave_height_m": h.get("wave_height", []),
            "wave_dir_deg": h.get("wave_direction", []),
            "wave_period_s": h.get("wave_period", []),
        })
        if len(df)==0: raise HTTPError("Hourly empty")
        return _time_filter(df), "Open-Meteo Marine (H+Dir+Per)"
    except Exception:
        # 2) fallback hauteur seule
        try:
            params = {"latitude": lat, "longitude": lon, "hourly": "wave_height", "timezone": tz, "forecast_days": 4}
            r = requests.get(base_url, params=params, timeout=20); r.raise_for_status()
            j = r.json(); h = j.get("hourly", {})
            df = pd.DataFrame({
                "datetime": pd.to_datetime(h.get("time", [])),
                "wave_height_m": h.get("wave_height", []),
            })
            if len(df)==0: raise HTTPError("Hourly empty (fallback)")
            df["wave_dir_deg"] = np.nan; df["wave_period_s"] = np.nan
            return _time_filter(df), "Open-Meteo Marine (hauteur seule – fallback)"
        except Exception:
            times = pd.date_range(now, now + timedelta(hours=hours), freq="H")
            df = pd.DataFrame({"datetime": pd.to_datetime(times), "wave_height_m": np.nan, "wave_dir_deg": np.nan, "wave_period_s": np.nan})
            return _time_filter(df), "Open-Meteo Marine indisponible (NaN)"

# -----------------------------
# ASTRONOMIE (API + Fallback)
# -----------------------------
def fetch_openmeteo_astronomy(lat, lon, tz="Europe/Paris", days=4):
    start = date.today(); end = (start + timedelta(days=days-1))
    url = "https://api.open-meteo.com/v1/astronomy"
    params = {"latitude": lat, "longitude": lon, "daily": "sunrise,sunset,moon_phase", "timezone": tz,
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    r = requests.get(url, params=params, timeout=20); r.raise_for_status()
    d = r.json().get("daily", {})
    if not d or "time" not in d: raise HTTPError("Astronomy daily block missing 'time'")
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]).dt.date,
        "sunrise": pd.to_datetime(d["sunrise"]),
        "sunset": pd.to_datetime(d["sunset"]),
        "moon_phase": d["moon_phase"],
    })
    df["sunrise"] = df["sunrise"].dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    df["sunset"]  = df["sunset"].dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    s = pd.Series(df["moon_phase"], dtype="float64")
    if s.max()>1.0: s = s/100.0
    df["moon_phase"] = s.clip(0,1)
    return df

def astronomy_fallback_astral(lat, lon, tz="Europe/Paris", days=4):
    try:
        from astral import LocationInfo
        from astral.sun import sun
        from astral.moon import phase as moon_phase_days
    except Exception as e:
        raise RuntimeError("Fallback astral indisponible : `pip install astral`") from e
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz)
    rows=[]
    for d in [date.today()+timedelta(days=i) for i in range(days)]:
        sdict = sun(loc.observer, date=d, tzinfo=LOCAL_TZ)
        sunrise, sunset = sdict["sunrise"], sdict["sunset"]
        days_since_new = float(moon_phase_days(d))  # 0..29.53
        f = (1 - math.cos(2*math.pi*(days_since_new/29.530588853)))/2.0
        rows.append({"date": d, "sunrise": sunrise, "sunset": sunset, "moon_phase": max(0.0,min(1.0,f))})
    return pd.DataFrame(rows)

def get_astronomy(lat, lon, tz="Europe/Paris", days=4):
    try:
        return fetch_openmeteo_astronomy(lat, lon, tz=tz, days=days), "Open-Meteo Astronomy"
    except (HTTPError, RequestException, KeyError, ValueError):
        return astronomy_fallback_astral(lat, lon, tz=tz, days=days), "Astral (fallback local)"

# -----------------------------
# WORLDTIDES (optionnel)
# -----------------------------
def fetch_worldtides_extremes(lat, lon, hours=HOURS, key: str | None = None):
    if not key: return [], "WorldTides: clé absente (bonus marée=0)"
    try:
        now = int(time.time()); length = int((hours + 12) * 3600)
        url = "https://www.worldtides.info/api/v3"
        params = {"extremes":"", "lat":lat, "lon":lon, "key":key, "start":now, "length":length}
        r = requests.get(url, params=params, timeout=20); r.raise_for_status()
        return r.json().get("extremes", []), "WorldTides"
    except Exception as e:
        return [], f"WorldTides erreur ({type(e).__name__}) – bonus marée=0"

def annotate_tides_on_hours(times: pd.Series, extremes: list[dict]) -> pd.DataFrame:
    if not len(extremes):
        return pd.DataFrame({"tide_state":[None]*len(times), "minutes_to_high":[np.nan]*len(times)})
    ex = sorted(extremes, key=lambda e: e["dt"])
    ex_times = [datetime.fromtimestamp(e["dt"], tz=LOCAL_TZ) for e in ex]
    ex_types = [e["type"] for e in ex]  # "High"/"Low"
    tide_state=[]; minutes_to_high=[]
    for t in times:
        diffs=[(abs((th-t).total_seconds()), th) for th, ty in zip(ex_times, ex_types) if ty.lower()=="high"]
        m_to_high=(min(diffs, key=lambda x:x[0])[1]-t).total_seconds()/60.0 if diffs else np.nan
        idx_next = next((i for i, th in enumerate(ex_times) if th>t), None)
        if idx_next is None or idx_next==0:
            state=None
        else:
            prev_type=ex_types[idx_next-1].lower(); next_type=ex_types[idx_next].lower()
            state = "rising" if (prev_type=="low" and next_type=="high") else ("falling" if (prev_type=="high" and next_type=="low") else None)
            if abs((ex_times[idx_next]-t).total_seconds())<=1800 or abs((ex_times[idx_next-1]-t).total_seconds())<=1800:
                state="slack"
        tide_state.append(state); minutes_to_high.append(m_to_high)
    return pd.DataFrame({"tide_state":tide_state, "minutes_to_high":minutes_to_high})

def find_next_high_tide(extremes: list[dict]) -> datetime | None:
    """Retourne la prochaine 'High' future en tz locale, sinon None."""
    if not extremes: return None
    now = datetime.now(LOCAL_TZ)
    highs = sorted([e for e in extremes if e.get("type","").lower()=="high"], key=lambda e: e.get("dt", 0))
    for e in highs:
        t = datetime.fromtimestamp(e["dt"], tz=LOCAL_TZ)
        if t > now: return t
    return None

# -----------------------------
# SST – COPERNICUS (prioritaire, multi-datasets)
# -----------------------------
def fetch_sst_copernicus_or_fallback(hours_df: pd.DataFrame, lat, lon):
    user = get_secret("COPERNICUS_USERNAME")
    pwd  = get_secret("COPERNICUS_PASSWORD")
    if not (user and pwd):
        return pd.Series([np.nan]*len(hours_df)), "SST indisponible (pas d'identifiants Copernicus)"

    try:
        from copernicusmarine import subset
        import xarray as xr

        dataset_candidates = [
            ("SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001-TDS",  ["analysed_sst","sst","sea_surface_temperature","analysis_sst","sstskin"]),
            ("SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",      ["analysed_sst","sst","sea_surface_temperature","analysis_sst","sstskin"]),
            ("SST_GLO_SST_L4_REP_OBSERVATIONS_010_011-TDS",  ["analysed_sst","sst","sea_surface_temperature"]),
            ("SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",      ["analysed_sst","sst","sea_surface_temperature"]),
        ]

        tmin = hours_df["datetime"].min().astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
        tmax = hours_df["datetime"].max().astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
        lon_min, lon_max = LON-0.10, LON+0.10
        lat_min, lat_max = LAT-0.10, LAT+0.10

        last_err = None

        for dsid, var_candidates in dataset_candidates:
            try:
                subset(
                    dataset_id=dsid,
                    variables=None,
                    minimum_longitude=lon_min, maximum_longitude=lon_max,
                    minimum_latitude=lat_min, maximum_latitude=lat_max,
                    start_datetime=tmin, end_datetime=tmax,
                    output_filename="sst_cmems.nc",
                    username=user, password=pwd,
                    force_download=True, overwrite_output_data=True
                )
                ds = xr.open_dataset("sst_cmems.nc")

                varname = None
                for v in var_candidates:
                    if v in ds.data_vars: varname = v; break
                if varname is None:
                    for v in ds.data_vars:
                        attrs = " ".join([f"{k}:{str(ds[v].attrs.get(k,''))}" for k in ds[v].attrs]).lower()
                        if "sst" in v.lower() or "sea surface temperature" in attrs:
                            varname = v; break
                if varname is None:
                    raise KeyError(f"Aucune variable SST reconnue dans {dsid}")

                sst = ds[varname]
                for d in ("lat","latitude","y"):
                    if d in sst.dims: sst = sst.mean(d, skipna=True)
                for d in ("lon","longitude","x"):
                    if d in sst.dims: sst = sst.mean(d, skipna=True)

                units = str(getattr(sst, "units", "")).lower()
                sst_vals = (sst - 273.15) if units in ("k","kelvin") else sst

                sst_df = sst_vals.to_dataframe(name="water_temp_c").reset_index()
                time_col = "time" if "time" in sst_df.columns else sst_df.columns[0]
                sst_df["datetime"] = pd.to_datetime(sst_df[time_col])
                if sst_df["datetime"].dt.tz is None:
                    sst_df["datetime"] = sst_df["datetime"].dt.tz_localize("UTC")
                sst_df["datetime"] = sst_df["datetime"].dt.tz_convert(LOCAL_TZ)

                sst_hourly = (sst_df[["datetime","water_temp_c"]]
                              .set_index("datetime").sort_index()
                              .resample("1H").interpolate().reset_index())

                merged = pd.merge_asof(
                    hours_df.sort_values("datetime"),
                    sst_hourly.sort_values("datetime"),
                    on="datetime", direction="nearest",
                    tolerance=pd.Timedelta("4H")
                )
                if merged["water_temp_c"].notna().any():
                    return merged["water_temp_c"], f"Copernicus Marine (dataset={dsid}, var={varname})"
                else:
                    last_err = f"{dsid}: valeurs NaN"
            except Exception as e:
                last_err = f"{dsid}: {type(e).__name__} – {e}"

        return pd.Series([np.nan]*len(hours_df)), f"Copernicus Marine indisponible ({last_err}) – SST NaN"

    except Exception as e:
        return pd.Series([np.nan]*len(hours_df)), f"Copernicus Marine indisponible ({type(e).__name__}: {e}) – SST NaN"

# -----------------------------
# SST – API LOCALE (FastAPI)
# -----------------------------
def fetch_sst_local_api(hours_df: pd.DataFrame):
    url = get_secret("LOCAL_SST_URL", "http://127.0.0.1:8000/api/temperature/eau")
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        temp = float(j.get("temperature"))
        if not (0.0 <= temp <= 30.0):
            raise ValueError("SST hors bornes plausibles")
        s = pd.Series(temp, index=hours_df.index, name="water_temp_c")
        return s, f"API locale SST ({url})"
    except Exception:
        return pd.Series([np.nan]*len(hours_df)), "API locale SST indisponible"

# -----------------------------
# SST – CABAIGNE.NET (scrape direct)
# -----------------------------
@st.cache_data(ttl=3*3600, show_spinner=False)
def scrape_cabaigne_sst() -> float | None:
    url = "https://www.cabaigne.net/france/normandie/havre/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        el = soup.find("div", class_="temperature")
        text = el.get_text(strip=True) if el else soup.get_text(" ", strip=True)
        m = re.search(r"(\d+[.,]?\d*)\s*°?\s*C", text, flags=re.IGNORECASE) or re.search(r"(\d+[.,]?\d*)", text)
        if not m: return None
        temp_c = float(m.group(1).replace(",", "."))
        return temp_c if 0.0 <= temp_c <= 30.0 else None
    except Exception:
        return None

def fetch_sst_cabaigne(hours_df: pd.DataFrame):
    val = scrape_cabaigne_sst()
    if val is None:
        return pd.Series([np.nan]*len(hours_df)), "cabaigne.net indisponible"
    return pd.Series(val, index=hours_df.index, name="water_temp_c"), "cabaigne.net (scrape)"

# -----------------------------
# SST – OPEN-METEO (water_temperature)
# -----------------------------
def fetch_openmeteo_sst_only(hours_df: pd.DataFrame, lat, lon, tz="Europe/Paris", hours=HOURS):
    base_url = "https://marine-api.open-meteo.com/v1/marine"
    now = datetime.now(LOCAL_TZ)
    try:
        params = {"latitude": lat, "longitude": lon, "hourly": "water_temperature", "timezone": tz, "forecast_days": 4}
        r = requests.get(base_url, params=params, timeout=20); r.raise_for_status()
        j = r.json(); h = j.get("hourly", {})
        if "time" not in h or "water_temperature" not in h:
            raise HTTPError("No water_temperature in hourly")
        sst_df = pd.DataFrame({"datetime": pd.to_datetime(h["time"]), "water_temp_c": h["water_temperature"]})
        if sst_df["datetime"].dt.tz is None:
            sst_df["datetime"] = sst_df["datetime"].dt.tz_localize(LOCAL_TZ)
        sst_df = sst_df[(sst_df["datetime"] >= now - timedelta(hours=1)) & (sst_df["datetime"] <= now + timedelta(hours=hours))].sort_values("datetime")
        merged = pd.merge_asof(hours_df.sort_values("datetime"), sst_df.sort_values("datetime"),
                               on="datetime", direction="nearest", tolerance=pd.Timedelta("2H"))
        return merged["water_temp_c"], "Open-Meteo Marine (water_temperature)"
    except Exception:
        return pd.Series([np.nan]*len(hours_df)), "Open-Meteo SST indisponible"

# -----------------------------
# ORCHESTRATION SST (priorité)
# -----------------------------
def fetch_sst_all_sources(hours_df: pd.DataFrame, lat, lon):
    # 1) Copernicus (multi-datasets)
    s1, src1 = fetch_sst_copernicus_or_fallback(hours_df, lat, lon)
    if s1.notna().any() and "indisponible" not in src1.lower():
        return s1, src1
    # 2) API locale FastAPI
    s2, src2 = fetch_sst_local_api(hours_df)
    if s2.notna().any() and "indisponible" not in src2.lower():
        return s2, src2
    # 3) cabaigne.net (scrape)
    s3, src3 = fetch_sst_cabaigne(hours_df)
    if s3.notna().any() and "indisponible" not in src3.lower():
        return s3, src3
    # 4) Open-Meteo water_temperature
    s4, src4 = fetch_openmeteo_sst_only(hours_df, lat, lon)
    if s4.notna().any() and "indisponible" not in src4.lower():
        return s4, src4
    # 5) Rien
    return pd.Series([np.nan]*len(hours_df)), "SST indisponible (Copernicus+API locale+cabaigne+Open-Meteo)"

# -----------------------------
# CHARGER & FUSIONNER LES SOURCES
# -----------------------------
@st.cache_data(ttl=900, show_spinner=True)
def load_all_sources():
    provenance = {}
    extras = {}  # infos additionnelles pour le récap (ex: prochaine PM)

    df_wx  = fetch_openmeteo_forecast(LAT, LON)
    provenance["Météo"] = "Open-Meteo Forecast"

    df_sea, sea_src = fetch_openmeteo_marine(LAT, LON)
    provenance["Houle"] = sea_src

    df_astro, astro_src = get_astronomy(LAT, LON)
    provenance["Astronomie"] = astro_src

    # merge meteo+mer
    df = pd.merge_asof(
        df_wx.sort_values("datetime"),
        df_sea.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("1H")
    )

    # SST (priorité & fallbacks)
    sst_series, sst_src = fetch_sst_all_sources(df, LAT, LON)
    df["water_temp_c"] = sst_series
    provenance["Température de l’eau (SST)"] = sst_src

    # Tendance pression 6h
    df = df.sort_values("datetime").reset_index(drop=True)
    df["pressure_trend_hpa_6h"] = np.nan
    for i in range(len(df)):
        t_now = df.loc[i,"datetime"]; t_prev = t_now - pd.Timedelta(hours=6)
        past = df[(df["datetime"] >= t_prev - pd.Timedelta(minutes=1)) & (df["datetime"] <= t_prev + pd.Timedelta(minutes=1))]
        if len(past) and pd.notnull(df.loc[i,"pressure_hpa"]) and pd.notnull(past.iloc[0]["pressure_hpa"]):
            df.loc[i,"pressure_trend_hpa_6h"] = df.loc[i]["pressure_hpa"] - past.iloc[0]["pressure_hpa"]

    # Marées
    wt_key = get_secret("WORLDTIDES_API_KEY", default=None)
    extremes, tide_src = fetch_worldtides_extremes(LAT, LON, key=wt_key)
    df_tide = annotate_tides_on_hours(df["datetime"], extremes)
    df = pd.concat([df, df_tide], axis=1)
    provenance["Marées"] = tide_src

    # Prochaine pleine mer (si extrêmes dispo)
    next_high = find_next_high_tide(extremes)
    extras["next_high_tide"] = next_high  # datetime ou None

    # Astro par ligne
    df["date"] = df["datetime"].dt.date
    df = df.merge(df_astro, on="date", how="left")

    return df, provenance, extras

df, provenance, extras = load_all_sources()

# -----------------------------
# SCORE BAR
# -----------------------------
def score_bar_row(row: pd.Series) -> float:
    ts = row["datetime"]
    ts_local = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
    if ts_local.tzinfo is None: ts_local = ts_local.tz_localize(LOCAL_TZ)

    s = 0.0
    s += tod_bonus_bar(ts_local, row.get("sunrise", None), row.get("sunset", None))
    s += temp_bonus_bar(row.get("water_temp_c", np.nan))
    s += wave_bonus_bar(row.get("wave_height_m", np.nan))
    s += wind_bonus_bar(row.get("wind_speed_ms", np.nan), row.get("wind_dir_deg", np.nan))
    s += pressure_bonus(row.get("pressure_hpa", np.nan), row.get("pressure_trend_hpa_6h", np.nan))
    s += tide_bonus(row.get("tide_state", None), row.get("minutes_to_high", np.nan))
    is_day = (row.get("sunrise", None) is not None and row.get("sunset", None) is not None and row["sunrise"] <= ts_local <= row["sunset"])
    s += cloud_bonus_bar(row.get("cloud_cover_pct", np.nan), is_day)
    s += moon_bonus(row.get("moon_phase", np.nan))
    return float(max(0.0, min(100.0, s)))

df["score_bar"] = df.apply(score_bar_row, axis=1)

if len(df)==0:
    st.error("Pas de données chargées depuis les APIs.")
    st.stop()

# -----------------------------
# PROVENANCE + ALERTES
# -----------------------------
st.subheader("🔎 Provenance des données utilisées")
st.markdown("\n".join([f"- **{k}** : {v}" for k, v in provenance.items()]))

if "SST indisponible" in provenance.get("Température de l’eau (SST)", ""):
    st.warning("🌡️ La température de l’eau n’a pas pu être récupérée (Copernicus/API locale/cabaigne/Open-Meteo).")

# -----------------------------
# RÉCAP MÉTÉO / MER / ASTRO / MARÉES
# -----------------------------
st.subheader("🧭 Récap rapide (maintenant)")
now_idx = 0
try:
    row = df.iloc[now_idx]
except Exception:
    row = None

colA, colB, colC, colD = st.columns(4)
with colA:
    try:
        st.metric("🌡️ Air (°C)", f"{row['air_temp_c']:.1f}")
        st.metric("🌊 Eau (°C)", f"{row['water_temp_c']:.1f}" if not np.isnan(row["water_temp_c"]) else "—")
    except Exception:
        st.metric("🌡️ Air (°C)", "—"); st.metric("🌊 Eau (°C)", "—")

with colB:
    try:
        st.metric("💨 Vent (m/s)", f"{row['wind_speed_ms']:.1f}")
        st.metric("↪️ Direction vent (°)", f"{row['wind_dir_deg']:.0f}")
    except Exception:
        st.metric("💨 Vent (m/s)", "—"); st.metric("↪️ Direction vent (°)", "—")

with colC:
    try:
        st.metric("📊 Pression (hPa)", f"{row['pressure_hpa']:.0f}")
        tr = row.get("pressure_trend_hpa_6h", np.nan)
        st.metric("Δ pression 6h (hPa)", f"{tr:+.1f}" if not np.isnan(tr) else "—")
    except Exception:
        st.metric("📊 Pression (hPa)", "—"); st.metric("Δ pression 6h (hPa)", "—")

with colD:
    try:
        st.metric("☁️ Nébulosité (%)", f"{row['cloud_cover_pct']:.0f}")
        st.metric("🌦️ Météo", code_to_desc(row.get("weathercode", None)))
    except Exception:
        st.metric("☁️ Nébulosité (%)", "—"); st.metric("🌦️ Météo", "—")

colE, colF, colG, colH = st.columns(4)
with colE:
    try:
        st.metric("🌊 Houle hauteur (m)", f"{row['wave_height_m']:.2f}" if not np.isnan(row["wave_height_m"]) else "—")
    except Exception:
        st.metric("🌊 Houle hauteur (m)", "—")
with colF:
    try:
        st.metric("🌊 Période (s)", f"{row['wave_period_s']:.1f}" if not np.isnan(row["wave_period_s"]) else "—")
    except Exception:
        st.metric("🌊 Période (s)", "—")
with colG:
    try:
        st.metric("🌅 Lever", row["sunrise"].strftime("%H:%M") if pd.notnull(row.get("sunrise", None)) else "—")
        st.metric("🌇 Coucher", row["sunset"].strftime("%H:%M") if pd.notnull(row.get("sunset", None)) else "—")
    except Exception:
        st.metric("🌅 Lever", "—"); st.metric("🌇 Coucher", "—")
with colH:
    nxt = extras.get("next_high_tide", None)
    if nxt:
        delta = nxt - datetime.now(LOCAL_TZ)
        hrs = int(delta.total_seconds()//3600); mins = int((delta.total_seconds()%3600)//60)
        st.metric("⛵ Prochaine pleine mer", f"{nxt:%a %d %b %H:%M}", f"Dans {hrs}h{mins:02d}")
    else:
        st.metric("⛵ Prochaine pleine mer", "—")

# -----------------------------
# TOP CRÉNEAUX + GRAPHS (Bar)
# -----------------------------
def top_windows(times, scores, min_score=60, merge_gap_minutes=120):
    out, group = [], None
    for t, s in zip(times, scores):
        if s >= min_score:
            group = [t, t] if group is None else [group[0], t]
        else:
            if group is not None: out.append(tuple(group)); group=None
    if group is not None: out.append(tuple(group))
    merged=[]
    for w in out:
        if not merged: merged.append(list(w))
        else:
            last=merged[-1]
            if (w[0]-last[1]).total_seconds()<=merge_gap_minutes*60:
                last[1]=w[1]
            else:
                merged.append(list(w))
    return [(a,b) for a,b in merged]

st.subheader("⭐ Top créneaux Bar + Graphiques (1/jour)")
df["_date"] = df["datetime"].dt.tz_convert(LOCAL_TZ).dt.date
days = sorted(df["_date"].unique())[:3]

if not len(days):
    st.info("Aucune donnée à tracer.")
else:
    for d in days:
        day_df = df[df["_date"] == d].copy()
        if day_df.empty: continue

        # Top créneaux
        st.markdown(f"**Top créneaux (≥ {min_score}) – {pd.to_datetime(d).strftime('%a %d %b')}**")
        wins = top_windows(day_df["datetime"], day_df["score_bar"].values, min_score=min_score)
        if not wins: st.write("Aucun créneau.")
        else:
            for a,b in wins: st.write(f"- {a:%H:%M} → {b:%H:%M}")

        # Graphique
        fig, ax = plt.subplots()
        ax.plot(day_df["datetime"], day_df["score_bar"], marker="o", linewidth=1.6, label="Bar")
        ax.set_title(f"Score Bar – {pd.to_datetime(d).strftime('%a %d %b %Y')} (Europe/Paris)")
        ax.set_xlabel("Heure"); ax.set_ylabel("Score /100")
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
        start = pd.Timestamp(d, tz=LOCAL_TZ); end = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        ax.set_xlim(start, end)
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper left"); plt.tight_layout()
        st.pyplot(fig)

# -----------------------------
# TABLE DES DONNÉES (optionnel)
# -----------------------------
if show_details:
    st.subheader("Données sources (heures)")
    st.dataframe(df[[
        "datetime","air_temp_c","pressure_hpa","pressure_trend_hpa_6h","wind_speed_ms","wind_dir_deg",
        "cloud_cover_pct","weathercode","wave_height_m","wave_dir_deg","wave_period_s",
        "water_temp_c","tide_state","minutes_to_high","sunrise","sunset",
        "moon_phase","score_bar"
    ]])

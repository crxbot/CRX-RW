#!/usr/bin/env python3
"""24h-Niederschlagssumme aus 24 RW-Stundendateien (RADOLAN RW).

Ausgabe: eine WebP-Datei (radarsumme_latest.webp)
  - Bild: leer/transparent (DRAW_IMAGE=False) oder farbige 24h-Summe (EPSG:3857)
  - Chunk 'RS24': sparse Zeitreihe, nur Pixel mit >= 0,1 mm in mindestens einer Stunde,
                  je Pixel alle 24 Stundenwerte (0,1-mm-Einheiten, int16)

Aufruf:
  python radarsumme24h.py                 # neueste RW-Datei als Ende des Fensters
  python radarsumme24h.py <rw-datei>      # bestimmte Datei als Ende
  python radarsumme24h.py --query LAT LON <webp>   # Zeitreihe abfragen
"""
import re
import struct
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/radarsumme")
OUT_DIR = Path("output/radarsumme24h")
OUT_FILENAME = "radarsumme24_latest.webp"   # wird bei jedem Lauf überschrieben
DRAW_IMAGE = False   # False: leeres transparentes Bild, nur der RS24-Chunk enthält Daten

# raa01-rw_10000-YYMMDDHHMM-dwd---bin.hdf5  (Zeit = Ende des 1h-Fensters, UTC)
FILENAME_RE = re.compile(r"raa01-rw_10000-(\d{10})-dwd---bin\.hdf5$")
HOURS = 24

DEFAULT_GAIN = 0.1
DEFAULT_NODATA = 65535

COLORS = [
    "#00C9FF", "#0057FF", "#0000EE", "#BEFFBD", "#98FE98", "#69FF68",
    "#30FF30", "#0AFF0A", "#00DC00", "#00BF00", "#008D00", "#FFFF00",
    "#F1D801", "#EABA00", "#F99C00", "#FE4100", "#FF2700", "#DC0000",
    "#B00000", "#FAC3FC", "#EBAAEA", "#DD95DE", "#C674C6", "#BA62B9",
    "#A342A3", "#861686", "#5C0F5C", "#410A41", "#320732",
]
LEVELS = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,
          35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350]
assert len(LEVELS) == len(COLORS) + 1

MAX_VALID_MM = 1000.0
MIN_VISIBLE_MM = LEVELS[0]

# Chunk
RS_FOURCC = b"RS24"
QUANTUM = 0.1            # mm pro int16-Einheit
STORE_MIN_UNITS = 1      # >= 0,1 mm wird gespeichert

MASK_OUTSIDE_RADAR = True
RADAR_RANGE_KM = 150.0
RADAR_MARGIN_KM = 0.0

RADARS = {
    "asb": dict(name="ASR Borkum",     lat=53.564011, lon=6.748292),
    "boo": dict(name="Boostedt",       lat=54.00438,  lon=10.04687),
    "drs": dict(name="Dresden",        lat=51.12465,  lon=13.76865),
    "eis": dict(name="Eisberg",        lat=49.54066,  lon=12.40278),
    "emd": dict(name="Emden",          lat=53.33872,  lon=7.02377),
    "ess": dict(name="Essen",          lat=51.40563,  lon=6.96712),
    "fbg": dict(name="Feldberg",       lat=47.87361,  lon=8.00361),
    "fld": dict(name="Flechtdorf",     lat=51.31120,  lon=8.802),
    "hnr": dict(name="Hannover",       lat=52.46008,  lon=9.69452),
    "neu": dict(name="Neuhaus",        lat=50.50012,  lon=11.13504),
    "nhb": dict(name="Neuheilenbach",  lat=50.10965,  lon=6.54853),
    "oft": dict(name="Offenthal",      lat=49.9847,   lon=8.71293),
    "pro": dict(name="Prötzel",        lat=52.64867,  lon=13.85821),
    "mem": dict(name="Memmingen",      lat=48.04214,  lon=10.21924),
    "ros": dict(name="Rostock",        lat=54.17566,  lon=12.05808),
    "isn": dict(name="Isen",           lat=48.17470,  lon=12.10177),
    "tur": dict(name="Türkheim",       lat=48.58528,  lon=9.78278),
    "umd": dict(name="Ummendorf",      lat=52.16009,  lon=11.17609),
}

WEBMERCATOR_OUT_WIDTH = 1927
EDGE_SAMPLES = 200
BBOX_MARGIN_DEG = 0.02
EARTH_RADIUS = 6378137.0


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_to_lonlat(x, y):
    lon = np.degrees(x / EARTH_RADIUS)
    lat = np.degrees(2 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2)
    return lon, lat


def parse_filename(filename: str) -> datetime:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Dateiname passt nicht zu 'raa01-rw_10000-YYMMDDHHMM-dwd---bin.hdf5': {filename}")
    return datetime.strptime(m.group(1), "%y%m%d%H%M").replace(tzinfo=timezone.utc)


TOLERANCE_MIN = 30   # Datei gilt für eine Stunde, wenn ihr Zeitstempel max. so weit abweicht


def index_files(src_dir: Path) -> dict[datetime, Path]:
    """Alle RW-Dateien im Ordner: Zeitstempel -> Pfad."""
    out = {}
    for p in src_dir.glob("raa01-rw_10000-*-dwd---bin.hdf5"):
        if FILENAME_RE.match(p.name):
            out[parse_filename(p.name)] = p
    return out


def find_hour_file(index: dict[datetime, Path], t: datetime) -> Path | None:
    """Datei, deren Zeitstempel am nächsten an t liegt (innerhalb TOLERANCE_MIN)."""
    best, best_diff = None, timedelta(minutes=TOLERANCE_MIN, seconds=1)
    for ts, p in index.items():
        diff = abs(ts - t)
        if diff < best_diff:
            best, best_diff = p, diff
    return best


def _attr_str(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def find_data_dataset(h5file: h5py.File) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError("Kein 2D-Datensatz in der Datei gefunden.")
    return candidates[0]


def read_sum_mm(ds: h5py.Dataset) -> np.ndarray:
    """Dataset -> mm (NaN = kein Wert)."""
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    attrs = what.attrs if what is not None else {}

    gain = float(_scalar(attrs.get("gain", DEFAULT_GAIN)))
    offset = float(_scalar(attrs.get("offset", 0.0)))
    nodata = _scalar(attrs.get("nodata", DEFAULT_NODATA))
    undetect = _scalar(attrs["undetect"]) if "undetect" in attrs else None

    raw = ds[()]
    values = raw.astype(np.float64) * gain + offset

    invalid = raw == nodata
    if np.issubdtype(raw.dtype, np.integer):
        invalid |= raw == np.iinfo(raw.dtype).max
    invalid |= ~np.isfinite(values)
    invalid |= values < 0
    invalid |= values > MAX_VALID_MM

    if undetect is not None and undetect != nodata:
        values[raw == undetect] = 0.0
        invalid &= raw != undetect
    values[invalid] = np.nan
    return values


def find_where_group(h5file: h5py.File):
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_float(key: str) -> float:
        return float(_scalar(where.attrs[key]))

    return {
        "projdef": _attr_str(where.attrs["projdef"]),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


def radar_coverage_mask(grid: dict, to_proj: Transformer) -> np.ndarray:
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    xs = ll_x + (np.arange(grid["xsize"]) + 0.5) * grid["xscale"]
    ys = ll_y + (grid["ysize"] - 1 - np.arange(grid["ysize"]) + 0.5) * grid["yscale"]
    xx, yy = np.meshgrid(xs, ys)

    limit = (RADAR_RANGE_KM + RADAR_MARGIN_KM) * 1000.0
    covered = np.zeros(xx.shape, dtype=bool)
    for info in RADARS.values():
        rx, ry = to_proj.transform(info["lon"], info["lat"])
        covered |= np.hypot(xx - rx, yy - ry) <= limit
    return covered


# --------------------------------------------------------------------------- #
# Geometrie / Warp
# --------------------------------------------------------------------------- #
def native_origin_and_extent(grid: dict, to_proj: Transformer):
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    x_max = ll_x + grid["xsize"] * grid["xscale"]
    y_max = ll_y + grid["ysize"] * grid["yscale"]
    return ll_x, ll_y, x_max, y_max


def wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84: Transformer):
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES)
    xs_span = ll_x + t * (x_max - ll_x)
    ys_span = ll_y + t * (y_max - ll_y)
    xs = np.concatenate([xs_span, xs_span, np.full_like(ys_span, ll_x), np.full_like(ys_span, x_max)])
    ys = np.concatenate([np.full_like(xs_span, ll_y), np.full_like(xs_span, y_max), ys_span, ys_span])
    lons, lats = (np.asarray(a) for a in to_wgs84.transform(xs, ys))
    return (
        float(lons.min()) - BBOX_MARGIN_DEG,
        float(lons.max()) + BBOX_MARGIN_DEG,
        float(lats.min()) - BBOX_MARGIN_DEG,
        float(lats.max()) + BBOX_MARGIN_DEG,
    )


def webmercator_target_grid(lon_min, lon_max, lat_min, lat_max):
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_h = max(int(round(WEBMERCATOR_OUT_WIDTH * aspect)), 1)
    x_new = np.linspace(x_min, x_max, WEBMERCATOR_OUT_WIDTH)
    y_new = np.linspace(y_min, y_max, out_h)
    return x_new, y_new, [float(x_min), float(y_min), float(x_max), float(y_max)]


def build_warp_map(grid, to_proj, x_new, y_new):
    """Einmal berechnen, dann für alle 24 Stunden wiederverwenden.
    Gibt (valid, row, col) zurück: Zielpixel -> Pixel im nativen Raster."""
    xx, yy = np.meshgrid(x_new, y_new)
    lon, lat = webmercator_to_lonlat(xx, yy)
    x_nat, y_nat = to_proj.transform(lon.ravel(), lat.ravel())
    x_nat = np.asarray(x_nat).reshape(xx.shape)
    y_nat = np.asarray(y_nat).reshape(xx.shape)

    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    col = np.floor((x_nat - ll_x) / grid["xscale"]).astype(np.int64)
    row = (grid["ysize"] - 1 - np.floor((y_nat - ll_y) / grid["yscale"])).astype(np.int64)
    valid = (col >= 0) & (col < grid["xsize"]) & (row >= 0) & (row < grid["ysize"])
    return valid, row[valid], col[valid]


def apply_warp(data, warp_map, fill=np.nan):
    valid, row, col = warp_map
    out = np.full(valid.shape, fill, dtype=np.float64)
    out[valid] = data[row, col]
    return out


# --------------------------------------------------------------------------- #
# Einfärben
# --------------------------------------------------------------------------- #
def colorize(total_mm: np.ndarray) -> np.ndarray:
    levels = np.array(LEVELS, dtype=np.float64)
    colors = np.array([hex_to_rgb(c) for c in COLORS], dtype=np.uint8)

    rgba = np.zeros((*total_mm.shape, 4), dtype=np.uint8)
    visible = np.isfinite(total_mm) & (total_mm >= MIN_VISIBLE_MM)
    if not visible.any():
        return rgba

    idx = np.searchsorted(levels - 1e-9, total_mm[visible], side="right") - 1
    idx = np.clip(idx, 0, len(colors) - 1)
    rgba[visible, :3] = colors[idx]
    rgba[visible, 3] = 255
    return rgba


# --------------------------------------------------------------------------- #
# RS24-Chunk (sparse Zeitreihe)
# --------------------------------------------------------------------------- #
def embed_sparse_chunk(webp_path: Path, stack: np.ndarray, stamps: list[int],
                       extent: list[float], quantum: float = QUANTUM) -> None:
    """stack: (T, H, W) int16, Zeile 0 = Norden, -1 = kein Wert."""
    t, h, w = stack.shape
    idx = np.flatnonzero((stack >= STORE_MIN_UNITS).any(axis=0).ravel())
    vals = np.ascontiguousarray(stack.reshape(t, -1)[:, idx].T)      # (N, T)

    deltas = np.diff(idx, prepend=0).astype("<u4")
    body = zlib.compress(deltas.tobytes() + vals.astype("<i2").tobytes(), 9)

    header = struct.pack("<BBII", 4, 4, w, h)
    header += struct.pack("<4d", *extent)
    header += struct.pack("<d", quantum)
    header += struct.pack("<H", t) + struct.pack(f"<{t}I", *stamps)
    header += struct.pack("<I", len(idx))
    payload = header + body

    chunk = RS_FOURCC + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        chunk += b"\x00"

    content = Path(webp_path).read_bytes()
    if content[:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise ValueError(f"{webp_path} ist keine gültige WebP-Datei")
    new_riff = struct.unpack("<I", content[4:8])[0] + len(chunk)
    Path(webp_path).write_bytes(content[:4] + struct.pack("<I", new_riff) + content[8:] + chunk)
    print(f"RS24: {len(idx)} von {h * w} Pixeln gespeichert, Chunk {len(chunk) / 1e6:.2f} MB")


def read_sparse(webp_path):
    d = Path(webp_path).read_bytes()
    pos = 12
    while pos < len(d):
        cc, size = d[pos:pos + 4], struct.unpack("<I", d[pos + 4:pos + 8])[0]
        if cc == RS_FOURCC:
            p = d[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size & 1)
    else:
        raise ValueError("kein RS24-Chunk")

    _, _, w, h = struct.unpack_from("<BBII", p, 0)
    extent = struct.unpack_from("<4d", p, 10)
    q, = struct.unpack_from("<d", p, 42)
    t, = struct.unpack_from("<H", p, 50)
    stamps = struct.unpack_from(f"<{t}I", p, 52)
    n, = struct.unpack_from("<I", p, 52 + 4 * t)
    raw = zlib.decompress(p[56 + 4 * t:])
    idx = np.cumsum(np.frombuffer(raw, "<u4", count=n).astype(np.int64))
    vals = np.frombuffer(raw, "<i2", offset=4 * n).reshape(n, t)
    return dict(w=w, h=h, extent=extent, q=q, stamps=stamps, idx=idx, vals=vals)


def query(s, lat, lon):
    xmin, ymin, xmax, ymax = s["extent"]
    x, y = lonlat_to_webmercator(lon, lat)
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return None
    col = int(round((x - xmin) / (xmax - xmin) * (s["w"] - 1)))
    row = int(round((ymax - y) / (ymax - ymin) * (s["h"] - 1)))
    flat = row * s["w"] + col

    k = np.searchsorted(s["idx"], flat)
    if k < len(s["idx"]) and s["idx"][k] == flat:
        v = s["vals"][k]
    else:
        v = np.zeros(len(s["stamps"]), dtype=np.int16)
    return [{"time": datetime.fromtimestamp(t_, timezone.utc).isoformat(),
             "mm": None if x_ < 0 else round(float(x_) * s["q"], 1)}
            for t_, x_ in zip(s["stamps"], v)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        lat, lon, path = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        res = query(read_sparse(path), lat, lon)
        if res is None:
            sys.exit("Koordinate außerhalb des Rasters.")
        for r in res:
            print(r["time"], r["mm"])
        return

    if len(sys.argv) > 1:
        end_path = Path(sys.argv[1])
        src_dir = end_path.parent
    else:
        src_dir = SRC_DIR
        candidates = sorted(p for p in src_dir.glob("raa01-rw_10000-*-dwd---bin.hdf5")
                            if FILENAME_RE.match(p.name))
        if not candidates:
            sys.exit(f"Keine RW-Datei in {src_dir} gefunden.")
        end_path = candidates[-1]

    index = index_files(src_dir)
    end_ts = parse_filename(end_path.name)
    times = [end_ts - timedelta(hours=h) for h in range(HOURS - 1, -1, -1)]   # älteste zuerst
    print(f"Fenster: {times[0]:%Y-%m-%d %H:%M} .. {times[-1]:%Y-%m-%d %H:%M} UTC ({HOURS} Stunden)")

    # Rasterinfo aus der ersten vorhandenen Datei
    grid = None
    for t in times:
        p = find_hour_file(index, t)
        if p is not None:
            with h5py.File(p, "r") as f:
                where = find_where_group(f)
                if where is None:
                    sys.exit("Keine 'where'-Projektionsinfo gefunden - Warp nicht möglich.")
                grid = extract_grid_info(where)
            break
    if grid is None:
        sys.exit("Keine der 24 RW-Dateien vorhanden.")

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    to_wgs84 = Transformer.from_crs(grid["projdef"], "EPSG:4326", always_xy=True)

    covered = radar_coverage_mask(grid, to_proj) if MASK_OUTSIDE_RADAR else None

    ll_x, ll_y, x_max, y_max = native_origin_and_extent(grid, to_proj)
    lon_min, lon_max, lat_min, lat_max = wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84)
    x_new, y_new, extent = webmercator_target_grid(lon_min, lon_max, lat_min, lat_max)
    print(f"EPSG:3857-Extent [xmin, ymin, xmax, ymax]: {extent}")
    print(f"Zielraster: {len(x_new)} x {len(y_new)} px")

    warp_map = build_warp_map(grid, to_proj, x_new, y_new)

    shape_native = (grid["ysize"], grid["xsize"])
    acc = np.zeros(shape_native, dtype=np.float64)       # Summe über vorhandene Stunden
    cnt = np.zeros(shape_native, dtype=np.int16)         # Anzahl gültiger Stunden je Pixel
    layers: list[np.ndarray] = []
    stamps: list[int] = []
    missing = []

    for t in times:
        stamps.append(int(t.timestamp()))
        layer = np.full((len(y_new), len(x_new)), -1, dtype=np.int16)
        p = find_hour_file(index, t)
        if p is None:
            missing.append(t)
            layers.append(layer[::-1])
            continue

        with h5py.File(p, "r") as f:
            mm = read_sum_mm(find_data_dataset(f))
        if mm.shape != shape_native:
            sys.exit(f"{p.name}: Rastergröße {mm.shape} passt nicht zu {shape_native}.")
        if covered is not None:
            mm = np.where(covered, mm, np.nan)

        ok = np.isfinite(mm)
        acc[ok] += mm[ok]
        cnt[ok] += 1

        merc = apply_warp(mm, warp_map)
        okm = np.isfinite(merc)
        layer[okm] = np.round(merc[okm] * 10).astype(np.int16)    # 0,1 mm
        layers.append(layer[::-1])                                # Norden oben
        print(f"  {t:%Y-%m-%d %H:%M} UTC  <-  {p.name}")

    if missing:
        print("WARNUNG: fehlende Stunden (Summe unvollständig): "
              + ", ".join(f"{m:%d.%m. %H:%M}" for m in missing), file=sys.stderr)

    # Bild: leer (voll transparent) oder farbige 24h-Summe
    if DRAW_IMAGE:
        total_native = np.where(cnt > 0, acc, np.nan)
        rgba = colorize(apply_warp(total_native, warp_map))
    else:
        rgba = np.zeros((len(y_new), len(x_new), 4), dtype=np.uint8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_FILENAME
    tmp_path = out_path.with_name(out_path.stem + ".tmp.webp")
    Image.fromarray(np.ascontiguousarray(rgba[::-1]), mode="RGBA").save(
        tmp_path, format="WEBP", lossless=True)

    stack = np.stack(layers)
    del layers
    embed_sparse_chunk(tmp_path, stack, stamps, extent)
    tmp_path.replace(out_path)    # atomar ersetzen, kein halbfertiges File für Leser
    print(f"Gespeichert: {out_path}  (Ende Fenster: {end_ts:%Y-%m-%d %H:%M} UTC)")


if __name__ == "__main__":
    main()
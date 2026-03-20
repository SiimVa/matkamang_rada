from typing import Optional
from urllib.parse import urlencode

try:
    import folium
except ImportError:
    folium = None

import geopandas as gpd
import matplotlib.pyplot as plt
import mgrs
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry import box

WFS_URL = "https://gsavalik.envir.ee/geoserver/wfs"
DEFAULT_MGRS = "\n".join(
    [
        "35VMD5794883765",
        "35VMD6859986372",
        "35VMD5491990072",
        "35VMD6508994336",
    ]
)
DEFAULT_LAYERS = {
    "seisuveekogud": True,
    "vooluveekogud": True,
    "margalad": True,
    "turbavaljad": False,
    "teed": True,
    "roobasteed": True,
    "sihid": True,
    "sillad": True,
    "eraomand": True,
}
ETAK_LAYER_MAP = {
    "seisuveekogud": "e_202_seisuveekogu_a",
    "vooluveekogud": "e_203_vooluveekogu_j",
    "margalad": "e_306_margala_a",
    "turbavaljad": "e_307_turbavali_a",
    "teed": "e_501_tee_j",
    "roobasteed": "e_502_roobastee_j",
    "sihid": "e_503_siht_j",
    "liiklusrajatised": "e_505_liikluskorralduslik_rajatis_ka",
}

m = mgrs.MGRS()
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3301", always_xy=True)


def mgrs_to_lest97(code: str) -> tuple[float, float]:
    lat, lon = m.toLatLon(code.strip())
    x, y = transformer.transform(lon, lat)
    return y, x


def parse_mgrs_codes(raw_value: str) -> list[str]:
    codes = [line.strip() for line in raw_value.splitlines() if line.strip()]
    unique_codes = list(dict.fromkeys(codes))
    if len(unique_codes) < 2:
        raise ValueError("Sisesta vähemalt kaks MGRS punkti.")
    return unique_codes


def build_bbox(codes: list[str]) -> dict:
    pts_yx = [mgrs_to_lest97(code) for code in codes]
    xs = [x for (y, x) in pts_yx]
    ys = [y for (y, x) in pts_yx]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bbox_geom = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:3301")
    return {
        "mgrs_codes": codes,
        "points_yx": pts_yx,
        "bbox_tuple": (minx, miny, maxx, maxy),
        "bbox_gdf": bbox_gdf,
    }


@st.cache_data(show_spinner=False)
def read_wfs_bbox(type_name: str, bbox_tuple: tuple[float, float, float, float], crs: str = "EPSG:3301") -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox_tuple
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "application/json",
        "srsName": crs,
        "bbox": f"{minx},{miny},{maxx},{maxy}",
    }
    url = f"{WFS_URL}?{urlencode(params)}"
    gdf = gpd.read_file(url, engine="pyogrio")

    if gdf.crs is None:
        gdf = gdf.set_crs(crs, allow_override=True)
    elif str(gdf.crs).upper() != crs.upper():
        gdf = gdf.to_crs(crs)

    clip_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=crs)
    return gpd.clip(gdf, clip_gdf)


def load_etak(layer_key: str, bbox_tuple: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    return read_wfs_bbox(f"etak:{ETAK_LAYER_MAP[layer_key]}", bbox_tuple)


def load_kataster(bbox_tuple: tuple[float, float, float, float], layer: str = "ky_kehtiv") -> gpd.GeoDataFrame:
    return read_wfs_bbox(f"kataster:{layer}", bbox_tuple)


def prepare_data(codes: list[str]) -> dict:
    bbox_info = build_bbox(codes)
    bbox_tuple = bbox_info["bbox_tuple"]

    data = {
        "bbox_gdf": bbox_info["bbox_gdf"],
        "bbox_tuple": bbox_tuple,
        "mgrs_codes": bbox_info["mgrs_codes"],
        "points_yx": bbox_info["points_yx"],
    }

    data["seisuveekogud"] = load_etak("seisuveekogud", bbox_tuple)
    data["vooluveekogud"] = load_etak("vooluveekogud", bbox_tuple)
    data["margalad"] = load_etak("margalad", bbox_tuple)
    data["turbavaljad"] = load_etak("turbavaljad", bbox_tuple)
    data["teed"] = load_etak("teed", bbox_tuple)
    data["roobasteed"] = load_etak("roobasteed", bbox_tuple)
    data["sihid"] = load_etak("sihid", bbox_tuple)
    data["liiklusrajatised"] = load_etak("liiklusrajatised", bbox_tuple)
    data["ky_clip"] = load_kataster(bbox_tuple)

    if "tyyp" in data["margalad"].columns:
        data["margalad"]["tyyp"] = pd.to_numeric(data["margalad"]["tyyp"], errors="coerce")
        data["margalad"] = data["margalad"][data["margalad"]["tyyp"].isin([10, 20])]

    liiklusrajatised = data["liiklusrajatised"]
    if "tyyp" in liiklusrajatised.columns:
        liiklusrajatised["tyyp"] = pd.to_numeric(liiklusrajatised["tyyp"], errors="coerce")
        sillad = liiklusrajatised[liiklusrajatised["tyyp"] == 30].copy()
    else:
        sillad = liiklusrajatised.iloc[0:0].copy()

    if not sillad.empty and not sillad.geom_type.isin(["Point", "MultiPoint"]).all():
        sillad["geometry"] = sillad.geometry.representative_point()

    data["sillad"] = sillad
    return data


def build_landscape_results(data: dict, layer_visibility: dict[str, bool]) -> pd.DataFrame:
    metrics = [
        ("seisuveekogud", "Seisuveekogud (km2)", data["seisuveekogud"].area.sum() / 1e6),
        ("vooluveekogud", "Vooluveekogud (km)", data["vooluveekogud"].length.sum() / 1000),
        ("margalad", "Märgalad (km2)", data["margalad"].area.sum() / 1e6),
        ("turbavaljad", "Turbaväljad (km2)", data["turbavaljad"].area.sum() / 1e6),
        ("teed", "Teed (km)", data["teed"].length.sum() / 1000),
        ("roobasteed", "Rööbasteed (km)", data["roobasteed"].length.sum() / 1000),
        ("sihid", "Sihid (km)", data["sihid"].length.sum() / 1000),
        ("sillad", "Sillad (arv)", float(len(data["sillad"]))),
    ]
    rows = []
    for layer_key, label, value in metrics:
        if not layer_visibility.get(layer_key, False):
            continue
        rows.append(
            {
                "Näitaja": label,
                "Väärtus": value,
                "Olemas": "Jah" if value > 0 else "Ei",
            }
        )
    return pd.DataFrame(rows)


def detect_private_owner(ky_clip: gpd.GeoDataFrame, target: str) -> tuple[Optional[str], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if target not in ky_clip.columns:
        empty = ky_clip.iloc[0:0].copy()
        return None, empty, ky_clip.copy()

    vals = ky_clip[target].dropna().astype(str).unique()
    era_values = [v for v in vals if "era" in v.lower()]

    if len(ky_clip) == 0:
        empty = ky_clip.iloc[0:0].copy()
        return None, empty, empty

    if len(era_values) == 0:
        era_value = ky_clip[target].dropna().astype(str).value_counts().idxmax() if ky_clip[target].notna().any() else None
    else:
        era_value = era_values[0]

    if era_value is None:
        empty = ky_clip.iloc[0:0].copy()
        return None, empty, ky_clip.copy()

    era = ky_clip[ky_clip[target] == era_value].copy()
    muu = ky_clip[ky_clip[target] != era_value].copy()
    return era_value, era, muu


def build_owner_tables(ky_clip: gpd.GeoDataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist_count = (
        ky_clip[target]
        .fillna("PUUDUB")
        .astype(str)
        .value_counts()
        .rename_axis("Omandivorm")
        .reset_index(name="Arv")
    )
    dist_count["Osakaal (%)"] = (dist_count["Arv"] / dist_count["Arv"].sum() * 100).round(2)

    if "pindala" in ky_clip.columns:
        ky_clip = ky_clip.copy()
        ky_clip["area_value"] = pd.to_numeric(ky_clip["pindala"], errors="coerce")
    else:
        ky_clip = ky_clip.copy()
        ky_clip["area_value"] = ky_clip.geometry.area

    dist_area = ky_clip.groupby(target, dropna=False)["area_value"].sum().reset_index()
    dist_area[target] = dist_area[target].fillna("PUUDUB").astype(str)
    dist_area["Pindala (km2)"] = dist_area["area_value"] / 1e6
    total_area = dist_area["area_value"].sum()
    if total_area:
        dist_area["Osakaal (%)"] = (dist_area["area_value"] / total_area * 100).round(2)
    else:
        dist_area["Osakaal (%)"] = 0.0
    dist_area = dist_area.rename(columns={target: "Omandivorm"}).drop(columns=["area_value"])

    return dist_count, dist_area


def format_analysis_summary(data: dict) -> str:
    area_km2 = data["bbox_gdf"].geometry.area.iloc[0] / 1e6
    mgrs_lines = "\n".join(f"{idx}. {code}" for idx, code in enumerate(data["mgrs_codes"], start=1))
    return f"MGRS punktid:\n{mgrs_lines}\n\nAla suurus: {area_km2:.2f} km2"


def get_layer_catalog() -> list[tuple[str, str, str, dict]]:
    return [
        ("seisuveekogud", "Seisuveekogud", "polygon", {"color": "blue", "fillColor": "lightblue", "weight": 1, "fillOpacity": 0.5}),
        ("vooluveekogud", "Vooluveekogud", "line", {"color": "blue", "weight": 2}),
        ("margalad", "Märgalad", "polygon", {"color": "darkgreen", "weight": 1, "fillColor": "#7fbf7b", "fillOpacity": 0.2}),
        ("turbavaljad", "Turbaväljad", "polygon", {"color": "#8c510a", "weight": 1, "fillColor": "#bf812d", "fillOpacity": 0.25}),
        ("teed", "Teed", "line", {"color": "black", "weight": 3}),
        ("roobasteed", "Rööbasteed", "line", {"color": "red", "weight": 2}),
        ("sihid", "Sihid", "line", {"color": "black", "weight": 2, "dashArray": "6, 4"}),
        ("sillad", "Sillad", "point", {}),
        ("eraomand", "Eraomand", "polygon", {"color": "red", "weight": 1, "fillColor": "#ff8080", "fillOpacity": 0.15}),
    ]


def build_osm_map(data: dict, era: gpd.GeoDataFrame, layer_visibility: dict[str, bool]) -> Optional[str]:
    if folium is None:
        return None

    bbox = data["bbox_gdf"].to_crs("EPSG:4326")
    bounds = bbox.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

    feature_sources = {
        "eraomand": era,
        "seisuveekogud": data["seisuveekogud"],
        "vooluveekogud": data["vooluveekogud"],
        "margalad": data["margalad"],
        "turbavaljad": data["turbavaljad"],
        "teed": data["teed"],
        "roobasteed": data["roobasteed"],
        "sihid": data["sihid"],
        "sillad": data["sillad"],
    }

    for layer_key, layer_label, geometry_type, style_kwargs in get_layer_catalog():
        if not layer_visibility.get(layer_key, False):
            continue

        gdf = feature_sources[layer_key]
        if gdf.empty:
            continue

        feature_group = folium.FeatureGroup(name=layer_label, show=True)
        gdf_wgs84 = gdf.to_crs("EPSG:4326")

        if geometry_type == "point":
            for geom in gdf_wgs84.geometry:
                if geom is None or geom.is_empty:
                    continue
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    radius=5,
                    color="black",
                    weight=1,
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=1,
                ).add_to(feature_group)
        else:
            folium.GeoJson(gdf_wgs84, style_function=lambda _feature, style=style_kwargs: style).add_to(feature_group)

        feature_group.add_to(fmap)

    folium.GeoJson(
        bbox,
        name="Analüüsiala",
        style_function=lambda _feature: {"color": "black", "weight": 2, "fillOpacity": 0},
    ).add_to(fmap)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: white;
        border: 1px solid black;
        padding: 10px 12px;
        font-size: 13px;
        line-height: 1.5;
    ">
      <strong>Legend</strong><br>
      <span style="display:inline-block;width:12px;height:12px;background:lightblue;border:1px solid blue;margin-right:6px;"></span>Seisuveekogud<br>
      <span style="display:inline-block;width:12px;height:2px;background:blue;margin-right:6px;vertical-align:middle;"></span>Vooluveekogud<br>
      <span style="display:inline-block;width:12px;height:12px;background:#7fbf7b;border:1px solid darkgreen;margin-right:6px;"></span>Märgalad<br>
      <span style="display:inline-block;width:12px;height:12px;background:#bf812d;border:1px solid #8c510a;margin-right:6px;"></span>Turbaväljad<br>
      <span style="display:inline-block;width:12px;height:2px;background:black;margin-right:6px;vertical-align:middle;"></span>Teed<br>
      <span style="display:inline-block;width:12px;height:2px;background:red;margin-right:6px;vertical-align:middle;"></span>Rööbasteed<br>
      <span style="display:inline-block;width:12px;height:2px;border-top:2px dashed black;margin-right:6px;vertical-align:middle;"></span>Sihid<br>
      <span style="display:inline-block;width:10px;height:10px;background:yellow;border:1px solid black;border-radius:50%;margin-right:6px;"></span>Sillad<br>
      <span style="display:inline-block;width:12px;height:12px;background:#ff8080;border:1px solid red;margin-right:6px;"></span>Eraomand
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return fmap.get_root().render()


def add_legend(ax, include_era: bool = False) -> None:
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.7, label="Teed"),
        Line2D([0], [0], color="black", lw=0.7, ls="--", label="Sihid"),
        Line2D([0], [0], color="red", lw=1.3, label="Rööbasteed"),
        Line2D([0], [0], color="blue", lw=1.0, label="Vooluveekogud"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markersize=8,
            label="Sillad",
        ),
        Patch(facecolor="lightblue", edgecolor="blue", alpha=0.7, label="Seisuveekogud"),
        Patch(facecolor="none", edgecolor="darkgreen", hatch="///", label="Märgalad"),
    ]

    if include_era:
        legend_handles.append(Patch(facecolor="none", edgecolor="red", hatch="///", label="Eraomand"))

    legend = ax.legend(handles=legend_handles, loc="lower right", title="Legend", frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1)
    frame.set_edgecolor("black")


def style_axes(ax, bbox_tuple: tuple[float, float, float, float], title: str) -> None:
    minx, miny, maxx, maxy = bbox_tuple
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)


def plot_analysis_map(data: dict, layer_visibility: dict[str, bool]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 9))

    if layer_visibility["margalad"] and not data["margalad"].empty:
        data["margalad"].plot(ax=ax, facecolor="none", edgecolor="darkgreen", hatch="///", linewidth=0.4, zorder=1)
    if layer_visibility["seisuveekogud"] and not data["seisuveekogud"].empty:
        data["seisuveekogud"].plot(ax=ax, facecolor="lightblue", edgecolor="blue", alpha=0.7, linewidth=0.3, zorder=2)
    if layer_visibility["vooluveekogud"] and not data["vooluveekogud"].empty:
        data["vooluveekogud"].plot(ax=ax, color="blue", linewidth=1.0, zorder=3)
    if layer_visibility["roobasteed"] and not data["roobasteed"].empty:
        data["roobasteed"].plot(ax=ax, color="red", linewidth=1.3, zorder=4)
    if layer_visibility["teed"] and not data["teed"].empty:
        data["teed"].plot(ax=ax, color="white", linewidth=2.2, zorder=9)
        data["teed"].plot(ax=ax, color="black", linewidth=1.7, zorder=10)
    if layer_visibility["sihid"] and not data["sihid"].empty:
        data["sihid"].plot(ax=ax, color="white", linewidth=2.0, zorder=11)
        data["sihid"].plot(ax=ax, color="black", linestyle="--", linewidth=0.7, zorder=12)
    if layer_visibility["sillad"] and not data["sillad"].empty:
        data["sillad"].plot(
            ax=ax,
            facecolor="yellow",
            edgecolor="black",
            marker="o",
            markersize=50,
            linewidth=0.8,
            zorder=13,
        )

    data["bbox_gdf"].boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=20)
    style_axes(ax, data["bbox_tuple"], "Analüüsiala")
    add_legend(ax, include_era=False)
    return fig


def plot_private_map(data: dict, era: gpd.GeoDataFrame, muu: gpd.GeoDataFrame, show_bridges: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))

    if not muu.empty:
        muu.boundary.plot(ax=ax, linewidth=0.3, color="black", zorder=1)
    if not era.empty:
        era.plot(ax=ax, facecolor="none", edgecolor="red", hatch="///", linewidth=0.6, zorder=2)
    if show_bridges and not data["sillad"].empty:
        data["sillad"].plot(
            ax=ax,
            facecolor="yellow",
            edgecolor="black",
            marker="o",
            markersize=50,
            linewidth=0.8,
            zorder=3,
        )

    data["bbox_gdf"].boundary.plot(ax=ax, color="red", linewidth=1.5, zorder=4)
    style_axes(ax, data["bbox_tuple"], "Eraomandid analüüsialas")

    legend_handles = [Patch(facecolor="none", edgecolor="red", hatch="///", label="Eraomand")]
    if show_bridges:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="yellow",
                markeredgecolor="black",
                markersize=8,
                label="Sillad",
            )
        )

    legend = ax.legend(handles=legend_handles, loc="lower right", title="Legend", frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1)
    frame.set_edgecolor("black")
    return fig


def plot_combined_map(data: dict, era: gpd.GeoDataFrame, layer_visibility: dict[str, bool]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 9))

    if layer_visibility["margalad"] and not data["margalad"].empty:
        data["margalad"].plot(ax=ax, facecolor="none", edgecolor="darkgreen", hatch="///", linewidth=0.4, zorder=1)
    if layer_visibility["seisuveekogud"] and not data["seisuveekogud"].empty:
        data["seisuveekogud"].plot(ax=ax, facecolor="lightblue", edgecolor="blue", alpha=0.7, linewidth=0.3, zorder=2)
    if layer_visibility["vooluveekogud"] and not data["vooluveekogud"].empty:
        data["vooluveekogud"].plot(ax=ax, color="blue", linewidth=1.0, zorder=3)
    if layer_visibility["roobasteed"] and not data["roobasteed"].empty:
        data["roobasteed"].plot(ax=ax, color="red", linewidth=1.3, zorder=4)
    if layer_visibility["teed"] and not data["teed"].empty:
        data["teed"].plot(ax=ax, color="white", linewidth=2.2, zorder=9)
        data["teed"].plot(ax=ax, color="black", linewidth=1.7, zorder=10)
    if layer_visibility["sihid"] and not data["sihid"].empty:
        data["sihid"].plot(ax=ax, color="white", linewidth=2.0, zorder=11)
        data["sihid"].plot(ax=ax, color="black", linestyle="--", linewidth=0.7, zorder=12)
    if layer_visibility["eraomand"] and not era.empty:
        era.plot(ax=ax, facecolor="none", edgecolor="red", hatch="///", linewidth=0.4, zorder=15)
    if layer_visibility["sillad"] and not data["sillad"].empty:
        data["sillad"].plot(
            ax=ax,
            facecolor="yellow",
            edgecolor="black",
            marker="o",
            markersize=50,
            linewidth=0.8,
            zorder=16,
        )

    data["bbox_gdf"].boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=20)
    style_axes(ax, data["bbox_tuple"], "Analüüsiala + eraomandid")
    add_legend(ax, include_era=layer_visibility["eraomand"])
    return fig


def main() -> None:
    st.set_page_config(page_title="Rada + katastri analüüs", layout="wide")
    st.title("Rada + katastri analüüs")
    st.caption("Maa- ja Ruumiameti WFS andmetel põhinev Streamliti rakendus.")

    with st.sidebar:
        st.header("Sisend")
        mgrs_input = st.text_area("MGRS punktid", key="mgrs_input", height=160, placeholder=DEFAULT_MGRS)
        st.caption("Sisesta üks MGRS kood reale. Vähemalt 2 punkti, soovitatavalt 4 BBOX moodustamiseks.")

        st.subheader("Kaardikihid")
        layer_visibility = {
            "seisuveekogud": st.checkbox("Seisuveekogud", value=DEFAULT_LAYERS["seisuveekogud"]),
            "vooluveekogud": st.checkbox("Vooluveekogud", value=DEFAULT_LAYERS["vooluveekogud"]),
            "margalad": st.checkbox("Märgalad", value=DEFAULT_LAYERS["margalad"]),
            "turbavaljad": st.checkbox("Turbaväljad", value=DEFAULT_LAYERS["turbavaljad"]),
            "teed": st.checkbox("Teed", value=DEFAULT_LAYERS["teed"]),
            "roobasteed": st.checkbox("Rööbasteed", value=DEFAULT_LAYERS["roobasteed"]),
            "sihid": st.checkbox("Sihid", value=DEFAULT_LAYERS["sihid"]),
            "sillad": st.checkbox("Sillad", value=DEFAULT_LAYERS["sillad"]),
            "eraomand": st.checkbox("Eraomand", value=DEFAULT_LAYERS["eraomand"]),
        }

        run_analysis = st.button("Käivita analüüs", type="primary", use_container_width=True)

    if run_analysis:
        try:
            codes = parse_mgrs_codes(mgrs_input)
            with st.spinner("Laen ETAK ja katastri andmeid..."):
                data = prepare_data(codes)
            st.session_state["analysis_data"] = data
        except Exception as exc:
            st.error(f"Andmete laadimine ebaõnnestus: {exc}")
            return

    data = st.session_state.get("analysis_data")
    if data is None:
        st.info("Muuda vasakul sisendeid ja käivita analüüs.")
        return

    if run_analysis:
        st.success("Analüüs valmis.")
    else:
        st.info("Kuvatud on viimati käivitatud analüüs. Kaardikihte saad muuta ilma tulemust kaotamata.")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Analüüsiala")
        st.code(format_analysis_summary(data))
    with col2:
        st.subheader("Aktiivsed kihid")
        active_layers = [label for key, label, _, _ in get_layer_catalog() if layer_visibility.get(key, False)]
        st.write(", ".join(active_layers) if active_layers else "Ühtegi kaardikihti pole valitud.")

    landscape_results = build_landscape_results(data, layer_visibility)
    st.subheader("Maastiku analüüs")
    if landscape_results.empty:
        st.info("Maastikuanalüüsi tabel on tühi, sest ükski vastav kaartkiht pole aktiivne.")
    else:
        st.dataframe(
            landscape_results.style.format({"Väärtus": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    target = "omvorm"
    if target in data["ky_clip"].columns and not data["ky_clip"].empty:
        dist_count, dist_area = build_owner_tables(data["ky_clip"], target)
        era_value, era, muu = detect_private_owner(data["ky_clip"], target)

        st.subheader("Omvormi jaotus")
        table_col1, table_col2 = st.columns(2)
        with table_col1:
            st.caption("Jaotus arvu järgi")
            st.dataframe(
                dist_count.style.format({"Osakaal (%)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
        with table_col2:
            st.caption("Jaotus pindala järgi")
            st.dataframe(
                dist_area.style.format({"Pindala (km2)": "{:.3f}", "Osakaal (%)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        if era_value:
            st.caption(f"Eraomand tuvastati väärtuse järgi: `{era_value}`")
    else:
        era = data["ky_clip"].iloc[0:0].copy()
        muu = data["ky_clip"].copy()
        st.warning("Katastri kihis puudub väli `omvorm`, seega omandivormi jaotust ei arvutatud.")

    st.subheader("Kaardid")
    tab1, tab2, tab3, tab4 = st.tabs(["Analüüsiala", "Eraomand", "Koondkaart", "OSM"])

    with tab1:
        fig = plot_analysis_map(data, layer_visibility)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig = plot_private_map(data, era, muu, layer_visibility["sillad"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        fig = plot_combined_map(data, era, layer_visibility)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab4:
        osm_html = build_osm_map(data, era, layer_visibility)
        if osm_html is None:
            st.warning("OSM kaart vajab paketti `folium`. Lisa see sõltuvustesse ja paigalda keskkonda.")
        else:
            components.html(osm_html, height=700)
            st.download_button(
                "Laadi OSM kaart alla HTML-na",
                data=osm_html,
                file_name="analuusi_kaart_osm.html",
                mime="text/html",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

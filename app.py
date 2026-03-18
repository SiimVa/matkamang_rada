from typing import Optional
from urllib.parse import urlencode

import geopandas as gpd
import matplotlib.pyplot as plt
import mgrs
import pandas as pd
import streamlit as st
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


def build_landscape_results(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "näitaja": [
                "Seisuveekogud km2",
                "Vooluveekogud km",
                "Märgalad km2",
                "Turbaväljad km2",
                "Teed km",
                "Rööbasteed km",
                "Sihid km",
                "Sillad arv",
            ],
            "väärtus": [
                data["seisuveekogud"].area.sum() / 1e6,
                data["vooluveekogud"].length.sum() / 1000,
                data["margalad"].area.sum() / 1e6,
                data["turbavaljad"].area.sum() / 1e6,
                data["teed"].length.sum() / 1000,
                data["roobasteed"].length.sum() / 1000,
                data["sihid"].length.sum() / 1000,
                len(data["sillad"]),
            ],
        }
    )


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
        .rename_axis("omvorm")
        .reset_index(name="arv")
    )
    dist_count["osakaal_%"] = (dist_count["arv"] / dist_count["arv"].sum() * 100).round(2)

    if "pindala" in ky_clip.columns:
        ky_clip = ky_clip.copy()
        ky_clip["pindala_m2"] = pd.to_numeric(ky_clip["pindala"], errors="coerce")
    else:
        ky_clip = ky_clip.copy()
        ky_clip["pindala_m2"] = ky_clip.geometry.area

    dist_area = ky_clip.groupby(target, dropna=False)["pindala_m2"].sum().reset_index()
    dist_area[target] = dist_area[target].fillna("PUUDUB").astype(str)
    dist_area["pindala_km2"] = dist_area["pindala_m2"] / 1e6
    dist_area["area_osakaal_%"] = (dist_area["pindala_m2"] / dist_area["pindala_m2"].sum() * 100).round(2)
    dist_area = dist_area.rename(columns={target: "omvorm"})

    return dist_count, dist_area


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
        mgrs_input = st.text_area("MGRS punktid", value=DEFAULT_MGRS, height=160)
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

    if not run_analysis:
        st.info("Muuda vasakul sisendeid ja käivita analüüs.")
        return

    try:
        codes = parse_mgrs_codes(mgrs_input)
        with st.spinner("Laen ETAK ja katastri andmeid..."):
            data = prepare_data(codes)
    except Exception as exc:
        st.error(f"Andmete laadimine ebaõnnestus: {exc}")
        return

    st.success("Analüüs valmis.")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Analüüsiala")
        minx, miny, maxx, maxy = data["bbox_tuple"]
        st.code(
            "\n".join(
                [
                    f"MGRS punktid (L-EST97, y/x): {data['points_yx']}",
                    f"BBOX: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})",
                ]
            )
        )
    with col2:
        st.subheader("Objektide arv")
        stats = pd.DataFrame(
            {
                "kiht": ["Veekogud", "Vooluveekogud", "Märgalad", "Turbaväljad", "Teed", "Rööbasteed", "Sihid", "Sillad", "Katastriüksused"],
                "objekte": [
                    len(data["seisuveekogud"]),
                    len(data["vooluveekogud"]),
                    len(data["margalad"]),
                    len(data["turbavaljad"]),
                    len(data["teed"]),
                    len(data["roobasteed"]),
                    len(data["sihid"]),
                    len(data["sillad"]),
                    len(data["ky_clip"]),
                ],
            }
        )
        st.dataframe(stats, use_container_width=True, hide_index=True)

    landscape_results = build_landscape_results(data)
    st.subheader("Maastiku analüüs")
    st.dataframe(
        landscape_results.style.format({"väärtus": "{:.2f}"}),
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
            st.dataframe(dist_count, use_container_width=True, hide_index=True)
        with table_col2:
            st.caption("Jaotus pindala järgi")
            st.dataframe(
                dist_area.style.format({"pindala_m2": "{:.0f}", "pindala_km2": "{:.3f}", "area_osakaal_%": "{:.2f}"}),
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
    tab1, tab2, tab3 = st.tabs(["Analüüsiala", "Eraomand", "Koondkaart"])

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


if __name__ == "__main__":
    main()

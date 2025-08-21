# app.py
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import plotly.express as px
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="폐렴 환자 대시보드", page_icon="🫁")
alt.themes.enable("dark")
st.title("요양기관 소재지 기준 폐렴 환자 대시보드")

# ─────────────────────────────────────────────
# 공용 유틸
# ─────────────────────────────────────────────
def series_to_df(s: pd.Series, value_name: str, index_name: str) -> pd.DataFrame:
    return s.to_frame(value_name).rename_axis(index_name).reset_index()

def build_region_gdf(geo_path: str) -> gpd.GeoDataFrame:
    """시·도 GeoJSON → 5개 권역 dissolve"""
    gdf = gpd.read_file(geo_path)

    # CRS 보정
    try:
        if gdf.crs is None:
            xmin, ymin, xmax, ymax = gdf.total_bounds
            if max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) > 200:
                gdf = gdf.set_crs(epsg=5179).to_crs(epsg=4326)
            else:
                gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass

    try:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.buffer(0)

    try:
        gdf = gdf.explode(index_parts=False)
    except Exception:
        gdf = gdf.explode()

    CODE_TO_REGION = {
        "11": "서울,인천", "28": "서울,인천",
        "41": "경기,강원", "42": "경기,강원",
        "43": "충청권(충북, 충남, 세종, 대전)", "44": "충청권(충북, 충남, 세종, 대전)",
        "30": "충청권(충북, 충남, 세종, 대전)", "36": "충청권(충북, 충남, 세종, 대전)",
        "45": "전라권(전북, 전남, 광주)", "46": "전라권(전북, 전남, 광주)", "29": "전라권(전북, 전남, 광주)",
        "47": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "48": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
        "26": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "27": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
        "31": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "50": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    }
    gdf["CTPRVN_CD"] = gdf["CTPRVN_CD"].astype(str).str.strip()
    gdf["권역"] = gdf["CTPRVN_CD"].map(CODE_TO_REGION)

    region_gdf = gdf.dropna(subset=["권역"]).dissolve(by="권역", as_index=False)[["권역", "geometry"]]
    return gpd.GeoDataFrame(region_gdf, geometry="geometry", crs=gdf.crs)

# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("데이터 선택")
    st.divider()

GEO_CANDIDATES = ["TL_SCCO_CTPRVN.json","data/TL_SCCO_CTPRVN.json"]
geo_path = next((p for p in GEO_CANDIDATES if Path(p).exists()), GEO_CANDIDATES[0])

data_file = "pneumonia_data.csv"
try:
    df_raw = pd.read_csv(data_file, encoding="utf-8-sig")
except FileNotFoundError:
    st.error(f"'{data_file}' 파일을 찾을 수 없습니다.")
    st.stop()

# ─────────────────────────────────────────────
# 전처리 & 공통 매핑
# ─────────────────────────────────────────────
df = df_raw.copy()
REGION_MAP = {
    1: "서울,인천", 2: "경기,강원",
    3: "충청권(충북, 충남, 세종, 대전)",
    4: "전라권(전북, 전남, 광주)",
    5: "경상권(경북, 경남, 부산, 대구, 울산, 제주)"
}
VALID_REGIONS = list(REGION_MAP.values())

def map_sex(s):
    s = str(s).strip()
    if s in ("1","남","male","Male","M","m"): return "남"
    if s in ("2","여","female","Female","F","f"): return "여"
    return np.nan

df["요양기관소재지_num"] = pd.to_numeric(df["요양기관소재지"], errors="coerce")
df["권역"] = df["요양기관소재지_num"].map(REGION_MAP)
df["성별_label"] = df["성별"].map(map_sex)
df = df[df["권역"].isin(VALID_REGIONS)].copy()

REGION_SIDO_N = {
    "서울,인천": 2, "경기,강원": 2,
    "충청권(충북, 충남, 세종, 대전)": 4,
    "전라권(전북, 전남, 광주)": 3,
    "경상권(경북, 경남, 부산, 대구, 울산, 제주)": 6,
}

# ─────────────────────────────────────────────
# 사이드바 필터
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("필터")
    if "요양기관종별" in df.columns:
        sel_types = st.multiselect("요양기관종별", sorted(df["요양기관종별"].astype(str).unique()), [])
        if sel_types:
            df = df[df["요양기관종별"].astype(str).isin(sel_types)]
    if "나이" in df.columns and len(df):
        age_min, age_max = int(df["나이"].min()), int(df["나이"].max())
        age_range = st.slider("나이 범위", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        df = df[(df["나이"] >= age_range[0]) & (df["나이"] <= age_range[1])]
    st.caption(f"현재 레코드 수: {len(df):,}")

# ─────────────────────────────────────────────
# 메인 페이지 (권역+지도+성별)
# ─────────────────────────────────────────────
st.subheader("권역별 분포 — 시도수 보정(표준화) 기준")

raw_counts = df["권역"].value_counts().reindex(VALID_REGIONS, fill_value=0)
std_counts = raw_counts.astype(float).div(pd.Series(REGION_SIDO_N))
std_pct = (std_counts / std_counts.sum() * 100).round(2)
plot_df = series_to_df(std_pct, "시도수 보정 비율(%)", "권역").sort_values("시도수 보정 비율(%)", ascending=False)

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("시도수 보정 비율(%):Q", title="비율(%)"),
            y=alt.Y("권역:N", sort="-x"),
            tooltip=["권역:N", "시도수 보정 비율(%):Q"],
            color=alt.Color("시도수 보정 비율(%):Q", scale=alt.Scale(scheme="reds"))
        ).properties(height=360)
    )
    text = (
        alt.Chart(plot_df)
        .mark_text(align="left", baseline="middle", dx=4)
        .encode(x="시도수 보정 비율(%):Q", y="권역:N", text="시도수 보정 비율(%):Q")
    )
    st.altair_chart(chart + text, use_container_width=True)

with c2:
    try:
        region_gdf = build_region_gdf(geo_path)
        map_df = region_gdf.merge(
            plot_df.rename(columns={"시도수 보정 비율(%)": "value"}),
            on="권역", how="left"
        ).fillna({"value": 0})
        geojson_obj = json.loads(map_df.to_json())
        vmax = float(map_df["value"].max()) if len(map_df) else 0.0

        fig_map = px.choropleth(
            data_frame=map_df.drop(columns=["geometry"]),
            geojson=geojson_obj,
            locations="권역",
            featureidkey="properties.권역",
            color="value",
            color_continuous_scale="OrRd",
            range_color=(0, vmax if vmax > 0 else 1),
            labels={"value": "시도수 보정 비율(%)"},
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0),
                              coloraxis_colorbar=dict(title="시도수 보정<br>비율(%)"))
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"지도를 생성하는 데 문제가 발생했습니다: {e}")

with st.expander("표(권역 시도수 보정 비율)"):
    st.dataframe(plot_df, use_container_width=True)

# ─────────────────────────────────────────────
# 성별 분석 (메인에 포함)
# ─────────────────────────────────────────────
st.markdown("### 성별 분석")

c3, c4 = st.columns([1, 2], gap="large")

gender = df[df["성별_label"].notna()]["성별_label"]
g_pct = (gender.value_counts(normalize=True) * 100).reindex(["남", "여"]).fillna(0).round(1)
g_df = series_to_df(g_pct, "비율(%)", "성별")

with c3:
    pie = px.pie(
        g_df, values="비율(%)", names="성별", color="성별",
        color_discrete_map={"남": "#66c2a5", "여": "#fc8d62"},
        hole=0.4, title="성별 분포(%)"
    )
    pie.update_traces(textinfo="label+percent")
    st.plotly_chart(pie, use_container_width=True)

with c4:
    cross = (
        df[df["성별_label"].notna()]
        .groupby(["권역", "성별_label"]).size()
        .groupby(level=0).apply(lambda s: s / s.sum() * 100)
        .reset_index(name="비율(%)")
    )
    bar = (
        alt.Chart(cross)
        .mark_bar()
        .encode(
            x=alt.X("비율(%):Q", title="비율(%)"),
            y=alt.Y("권역:N", sort="-x"),
            color=alt.Color("성별_label:N", title="성별",
                            scale=alt.Scale(domain=["남", "여"], range=["#66c2a5", "#fc8d62"])),
            tooltip=["권역:N", "성별_label:N", "비율(%):Q"]
        ).properties(title="권역별 성별 비율(권역 내 %)", height=320)
    )
    st.altair_chart(bar, use_container_width=True)

# ─────────────────────────────────────────────
# 풋노트
# ─────────────────────────────────────────────
st.caption("ⓒ Respiratory Rehab / Pneumonia Insights — 권역은 요양기관 소재지 기준, "
           "권역 막대그래프는 시도수 보정(시도당 평균) 후 100% 정규화한 비율을 사용합니다.")

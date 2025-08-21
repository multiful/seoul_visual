# app.py
import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="폐렴 환자 대시보드", page_icon="🫁")
alt.themes.enable("dark")
st.title("요양기관 소재지 기준 폐렴 환자 대시보드")

# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("데이터 선택")
    st.divider()

# 시·도 GeoJSON (업로드 파일 우선)
GEO_CANDIDATES = ["TL_SCCO_CTPRVN.json"]
geo_path = next((p for p in GEO_CANDIDATES if Path(p).exists()), GEO_CANDIDATES[0])

# 폐렴 데이터 CSV
data_file = "pneumonia_data.csv"
try:
    df_raw = pd.read_csv(data_file, encoding="utf-8-sig")
except FileNotFoundError:
    st.error("pneumonia_data.csv 파일을 찾을 수 없습니다.")
    st.stop()

# ─────────────────────────────────────────────
# 전처리 & 공통 매핑
# ─────────────────────────────────────────────
df = df_raw.copy()

REGION_MAP = {
    1: "서울,인천",
    2: "경기,강원",
    3: "충청권(충북, 충남, 세종, 대전)",
    4: "전라권(전북, 전남, 광주)",
    5: "경상권(경북, 경남, 부산, 대구, 울산, 제주)"
}
VALID_REGIONS = list(REGION_MAP.values())

def map_sex(s):
    s = str(s).strip()
    if s in ("1", "남", "male", "Male", "M", "m"):
        return "남"
    if s in ("2", "여", "female", "Female", "F", "f"):
        return "여"
    return np.nan

df["요양기관소재지_num"] = pd.to_numeric(df["요양기관소재지"], errors="coerce")
df["권역"] = df["요양기관소재지_num"].map(REGION_MAP)
df["성별_label"] = df["성별"].map(map_sex)

df = df[df["권역"].isin(VALID_REGIONS)].copy()

REGION_SIDO_N = {
    "서울,인천": 2,
    "경기,강원": 2,
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
    if "나이" in df.columns:
        age_min, age_max = int(df["나이"].min()), int(df["나이"].max())
        age_range = st.slider("나이 범위", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        df = df[(df["나이"] >= age_range[0]) & (df["나이"] <= age_range[1])]
    st.caption(f"현재 레코드 수: {len(df):,}")

# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab_main, tab_gender, tab_map, tab_corr = st.tabs(["메인", "성별 분석", "지도(권역)", "고령인구비율 상관"])

# ─────────────────────────────────────────────
# 메인: 권역 표준화(시도수 보정)
# ─────────────────────────────────────────────
with tab_main:
    st.subheader("권역별 분포 — 시도수 보정(표준화) 기준")

    raw_counts = df["권역"].value_counts().reindex(VALID_REGIONS, fill_value=0)
    std_counts = raw_counts.astype(float).div(pd.Series(REGION_SIDO_N))
    std_pct = (std_counts / std_counts.sum() * 100).round(2)

    # 🔧 문제되던 곳: rename().reset_index() 금지
    plot_df = std_pct.to_frame("시도수 보정 비율(%)").rename_axis("권역").reset_index()
    plot_df = plot_df.sort_values("시도수 보정 비율(%)", ascending=False)

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("시도수 보정 비율(%):Q", title="비율(%)"),
            y=alt.Y("권역:N", sort="-x"),
            tooltip=["권역:N", "시도수 보정 비율(%):Q"],
            color=alt.Color("시도수 보정 비율(%):Q", scale=alt.Scale(scheme="reds"))
        )
        .properties(height=320)
    )
    text = (
        alt.Chart(plot_df)
        .mark_text(align="left", baseline="middle", dx=4)
        .encode(x="시도수 보정 비율(%):Q", y="권역:N", text="시도수 보정 비율(%):Q")
    )
    st.altair_chart(chart + text, use_container_width=True)

    with st.expander("표 보기"):
        st.dataframe(plot_df, use_container_width=True)

# ─────────────────────────────────────────────
# 성별 분석
# ─────────────────────────────────────────────
with tab_gender:
    st.subheader("성별 분포(전체) 및 권역별 성별 비교")

    gender = df[df["성별_label"].notna()]["성별_label"]
    g_pct = (gender.value_counts(normalize=True) * 100).reindex(["남", "여"]).fillna(0).round(1)
    g_df = g_pct.to_frame("비율(%)").rename_axis("성별").reset_index()

    c1, c2 = st.columns([1, 2])
    with c1:
        pie = px.pie(g_df, values="비율(%)", names="성별", color="성별",
                     color_discrete_map={"남": "#66c2a5", "여": "#fc8d62"},
                     hole=0.4, title="성별 분포(%)")
        pie.update_traces(textinfo="label+percent")
        st.plotly_chart(pie, use_container_width=True)

    with c2:
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
            )
            .properties(title="권역별 성별 비율(권역 내 %)", height=320)
        )
        st.altair_chart(bar, use_container_width=True)

# ─────────────────────────────────────────────
# 지도: 권역 단위 Choropleth
# ─────────────────────────────────────────────
with tab_map:
    st.subheader("권역별 Choropleth — 시도수 보정 비율(%)")

    # GeoJSON(시·도) 로드
    try:
        gdf = gpd.read_file(geo_path)  # columns: CTPRVN_CD, CTP_KOR_NM, geometry ...
    except Exception as e:
        st.error(f"시·도 GeoJSON을 읽는 중 오류: {e}")
        st.stop()

    # CRS → EPSG:4326
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

    # 도형 유효화
    try:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.buffer(0)

    # 멀티폴리곤 분해
    try:
        gdf = gdf.explode(index_parts=False)
    except Exception:
        gdf = gdf.explode()

    xmin, ymin, xmax, ymax = gdf.total_bounds

    CODE_TO_REGION = {
        "11": "서울,인천", "28": "서울,인천",
        "41": "경기,강원", "42": "경기,강원",
        "43": "충청권(충북, 충남, 세종, 대전)", "44": "충청권(충북, 충남, 세종, 대전)",
        "36": "충청권(충북, 충남, 세종, 대전)", "30": "충청권(충북, 충남, 세종, 대전)",
        "45": "전라권(전북, 전남, 광주)", "46": "전라권(전북, 전남, 광주)", "29": "전라권(전북, 전남, 광주)",
        "47": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "48": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
        "26": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "27": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
        "31": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "50": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    }
    gdf["CTPRVN_CD"] = gdf["CTPRVN_CD"].astype(str).str.strip()
    gdf["권역"] = gdf["CTPRVN_CD"].map(CODE_TO_REGION)

    # 표준화 비율(지도용) — 안전한 생성
    std_counts_map = (
        df["권역"].value_counts()
        .reindex(VALID_REGIONS, fill_value=0)
        .astype(float)
        .div(pd.Series(REGION_SIDO_N))
    )
    std_pct_map = (std_counts_map / std_counts_map.sum() * 100).round(2)
    std_pct_df = std_pct_map.to_frame("비율(%)").rename_axis("권역").reset_index()

    # 권역 단위 디졸브 + 병합
    region_gdf = gdf.dissolve(by="권역", as_index=False)[["권역", "geometry"]]
    region_gdf = region_gdf.merge(std_pct_df, on="권역", how="left").fillna({"비율(%)": 0})
    region_gdf = gpd.GeoDataFrame(region_gdf, geometry="geometry", crs=gdf.crs)

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 10))
    region_gdf.plot(
        ax=ax, column="비율(%)", cmap="OrRd", legend=True,
        edgecolor="#333333", linewidth=0.6,
        legend_kwds={"shrink": 0.75, "orientation": "vertical"}
    )
    gdf.boundary.plot(ax=ax, color="#444444", linewidth=0.25, alpha=0.7)

    # 라벨
    try:
        for _, r in region_gdf.dropna(subset=["geometry"]).iterrows():
            p = r["geometry"].representative_point()
            ax.text(p.x, p.y, f"{r['비율(%)']:.1f}%", ha="center", va="center", fontsize=8, color="#1a1a1a")
    except Exception:
        pass

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box"); ax.margins(0)
    ax.set_title("요양기관 소재지 권역별 비율(시도수 보정)", fontsize=13)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────
# 고령인구비율 상관
# ─────────────────────────────────────────────
with tab_corr:
    st.subheader("고령인구비율과의 관계")
    st.caption("고정된 고령인구비율 파일을 사용해 권역 표준화 비율과의 관계를 봅니다.")

    xlsx_path = "고령인구비율_시도_시_군_구__20250821041330.xlsx"
    try:
        xls = pd.ExcelFile(xlsx_path)
        sheet = next((s for s in xls.sheet_names if "데이터" in s or "data" in s.lower()), xls.sheet_names[0])
        age_df = pd.read_excel(xlsx_path, sheet_name=sheet)

        sido_col_candidates = [c for c in age_df.columns if "행정구역" in str(c)]
        sido_col = sido_col_candidates[0] if sido_col_candidates else age_df.columns[0]

        year_cols = [c for c in age_df.columns if isinstance(c, (int, np.integer))]
        latest_year = max(year_cols)

        age_sido = age_df[[sido_col, latest_year]].rename(
            columns={sido_col: "시도", latest_year: "고령인구비율(%)"}
        )
        age_sido["시도"] = age_sido["시도"].replace({
            "강원도": "강원특별자치도",
            "전라북도": "전북특별자치도"
        })

        raw_counts2 = df["권역"].value_counts().reindex(VALID_REGIONS, fill_value=0)
        std_counts2 = raw_counts2.astype(float).div(pd.Series(REGION_SIDO_N))
        std_pct2 = (std_counts2 / std_counts2.sum() * 100)

        REGION_TO_SIDO = {
            "서울,인천": ["서울특별시", "인천광역시"],
            "경기,강원": ["경기도", "강원특별자치도"],
            "충청권(충북, 충남, 세종, 대전)": ["충청북도", "충청남도", "세종특별자치시", "대전광역시"],
            "전라권(전북, 전남, 광주)": ["전북특별자치도", "전라남도", "광주광역시"],
            "경상권(경북, 경남, 부산, 대구, 울산, 제주)": [
                "경상북도", "경상남도", "부산광역시", "대구광역시", "울산광역시", "제주특별자치도"
            ],
        }

        expand_rows = []
        for reg, pct in std_pct2.items():
            for s in REGION_TO_SIDO.get(reg, []):
                expand_rows.append({"시도": s, "표준화비율(%)": pct})
        reg_to_sido_df = pd.DataFrame(expand_rows)

        merged = pd.merge(reg_to_sido_df, age_sido, on="시도", how="inner")
        merged = merged.groupby("시도", as_index=False).mean(numeric_only=True)

        st.write("미리보기")
        st.dataframe(merged, use_container_width=True)

        corr = merged[["표준화비율(%)", "고령인구비율(%)"]].corr().iloc[0, 1]
        st.markdown(f"**상관계수 (권역 표준화비율 vs 시도 고령인구비율)**: `{corr:.2f}`")

        sc = alt.Chart(merged).mark_circle(size=120).encode(
            x=alt.X("고령인구비율(%):Q"),
            y=alt.Y("표준화비율(%):Q"),
            tooltip=["시도", "고령인구비율(%)", "표준화비율(%)"],
            color=alt.value("#e74c3c")
        )
        reg_line = sc.transform_regression("고령인구비율(%)", "표준화비율(%)").mark_line(color="#f39c12")
        st.altair_chart(sc + reg_line, use_container_width=True)

    except Exception as e:
        st.error(f"고령인구비율 파일 처리 중 오류: {e}")
        st.stop()

# ─────────────────────────────────────────────
# 풋노트
# ─────────────────────────────────────────────
st.caption("ⓒ Respiratory Rehab / Pneumonia Insights — 권역은 요양기관 소재지 기준, "
           "권역 막대그래프는 시도수 보정(시도당 평균) 후 100% 정규화한 비율을 사용합니다.")

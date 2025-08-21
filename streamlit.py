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
    """Series → DataFrame 변환 시 index/column 충돌 없이 안전하게 변환"""
    return s.to_frame(value_name).rename_axis(index_name).reset_index()

def build_region_gdf(geo_path: str) -> gpd.GeoDataFrame:
    """
    시·도 GeoJSON을 읽어 5개 권역으로 dissolve한 GeoDataFrame 반환
    (메인 탭 Plotly Choropleth에서 사용)
    """
    gdf = gpd.read_file(geo_path)  # must contain: CTPRVN_CD, geometry

    # CRS 보정 → EPSG:4326
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

    # 도형 유효화 + 멀티폴리곤 분해
    try:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.buffer(0)

    try:
        gdf = gdf.explode(index_parts=False)
    except Exception:
        gdf = gdf.explode()

    # 시도코드 → 권역 매핑 (충청 전부 포함 보장)
    CODE_TO_REGION = {
        # 수도권/인천
        "11": "서울,인천",  # 서울
        "28": "서울,인천",  # 인천
        # 경기·강원
        "41": "경기,강원",  # 경기
        "42": "경기,강원",  # 강원특별자치도
        # 충청권
        "43": "충청권(충북, 충남, 세종, 대전)",  # 충북
        "44": "충청권(충북, 충남, 세종, 대전)",  # 충남
        "30": "충청권(충북, 충남, 세종, 대전)",  # 대전
        "36": "충청권(충북, 충남, 세종, 대전)",  # 세종
        # 전라권
        "45": "전라권(전북, 전남, 광주)",  # 전북특별자치도
        "46": "전라권(전북, 전남, 광주)",  # 전남
        "29": "전라권(전북, 전남, 광주)",  # 광주
        # 경상권(+제주)
        "47": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",  # 경북
        "48": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",  # 경남
        "26": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",  # 부산
        "27": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",  # 대구
        "31": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",  # 울산
        "50": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",  # 제주
    }
    gdf["CTPRVN_CD"] = gdf["CTPRVN_CD"].astype(str).str.strip()
    gdf["권역"] = gdf["CTPRVN_CD"].map(CODE_TO_REGION)

    # 필수 권역만 남기고 dissolve
    region_gdf = gdf.dropna(subset=["권역"]).dissolve(by="권역", as_index=False)[["권역", "geometry"]]
    region_gdf = gpd.GeoDataFrame(region_gdf, geometry="geometry", crs=gdf.crs)

    # 누락 체크(디버그)
    expected = {
        "서울,인천",
        "경기,강원",
        "충청권(충북, 충남, 세종, 대전)",
        "전라권(전북, 전남, 광주)",
        "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    }
    got = set(region_gdf["권역"].unique())
    missing = expected - got
    if missing:
        st.warning(f"지오메트리에서 누락된 권역: {sorted(missing)} — CODE_TO_REGION 또는 원본 GeoJSON을 확인하세요.")

    return region_gdf

# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("데이터 선택")
    st.divider()

# GeoJSON(시·도) 경로
GEO_CANDIDATES = [
    "TL_SCCO_CTPRVN.json",
    "data/TL_SCCO_CTPRVN.json",
]
geo_path = next((p for p in GEO_CANDIDATES if Path(p).exists()), GEO_CANDIDATES[0])

# 폐렴 데이터 CSV 경로
data_file = "pneumonia_data.csv"

# 데이터 불러오기
try:
    df_raw = pd.read_csv(data_file, encoding="utf-8-sig")
except FileNotFoundError:
    st.error(f"'{data_file}' 파일을 찾을 수 없습니다.")
    st.stop()

# ─────────────────────────────────────────────
# 전처리 & 공통 매핑
# ─────────────────────────────────────────────
df = df_raw.copy()

# 권역 매핑 (요양기관소재지: 1~5)
REGION_MAP = {
    1: "서울,인천",
    2: "경기,강원",
    3: "충청권(충북, 충남, 세종, 대전)",
    4: "전라권(전북, 전남, 광주)",
    5: "경상권(경북, 경남, 부산, 대구, 울산, 제주)"
}
VALID_REGIONS = list(REGION_MAP.values())

# 성별 매핑
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

# 분석 대상만 남기기
df = df[df["권역"].isin(VALID_REGIONS)].copy()

# 시도수(권역별) - 표준화용
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
    if "나이" in df.columns and len(df):
        age_min, age_max = int(df["나이"].min()), int(df["나이"].max())
        age_range = st.slider("나이 범위", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        df = df[(df["나이"] >= age_range[0]) & (df["나이"] <= age_range[1])]
    st.caption(f"현재 레코드 수: {len(df):,}")

# ─────────────────────────────────────────────
# 탭 (지도가 메인에 포함됨)
# ─────────────────────────────────────────────
tab_main, tab_gender, tab_corr = st.tabs(["메인", "성별 분석", "고령인구비율 상관"])

# ─────────────────────────────────────────────
# 메인: 막대 + Plotly Choropleth(권역)
# ─────────────────────────────────────────────
with tab_main:
    st.subheader("권역별 분포 — 시도수 보정(표준화) 기준")

    # 표준화 비율 계산 (누락 방지 reindex)
    raw_counts = df["권역"].value_counts().reindex(VALID_REGIONS, fill_value=0)
    std_counts = raw_counts.astype(float).div(pd.Series(REGION_SIDO_N))
    std_pct = (std_counts / std_counts.sum() * 100).round(2)

    plot_df = (
        series_to_df(std_pct, "시도수 보정 비율(%)", "권역")
        .sort_values("시도수 보정 비율(%)", ascending=False)
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("시도수 보정 비율(%):Q", title="비율(%)"),
                y=alt.Y("권역:N", sort="-x"),
                tooltip=["권역:N", "시도수 보정 비율(%):Q"],
                color=alt.Color("시도수 보정 비율(%):Q", scale=alt.Scale(scheme="reds"))
            )
            .properties(height=380)
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

            # 지도에 들어있는 권역과 값 테이블 키 일치 여부 확인
            geo_regions = set(region_gdf["권역"].unique())
            val_regions = set(plot_df["권역"].unique())
            if geo_regions - val_regions:
                st.info(f"값 테이블에 없는 권역이 지오메트리에 있습니다: {sorted(geo_regions - val_regions)}")
            if val_regions - geo_regions:
                st.info(f"지오메트리에 없는 권역이 값 테이블에 있습니다: {sorted(val_regions - geo_regions)}")

            # 표준화 비율 병합 (권역 키)
            map_df = region_gdf.merge(
                plot_df.rename(columns={"시도수 보정 비율(%)": "value"}),
                on="권역", how="left"
            ).fillna({"value": 0})

            # GeoJSON 직렬화 (properties에 '권역' 포함)
            geojson_obj = json.loads(map_df.to_json())

            vmax = float(map_df["value"].max()) if len(map_df) else 0.0

            # Plotly Choropleth
            fig_map = px.choropleth(
                data_frame=map_df.drop(columns=["geometry"]),
                geojson=geojson_obj,
                locations="권역",
                featureidkey="properties.권역",
                color="value",
                color_continuous_scale="OrRd",
                range_color=(0, vmax if vmax > 0 else 1),
                labels={"value": "시도수 보정\n비율(%)"},
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                height=420, margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar=dict(title="시도수 보정<br>비율(%)")
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.warning(f"지도를 생성하는 데 문제가 발생했습니다: {e}")

    with st.expander("표 보기"):
        st.dataframe(plot_df, use_container_width=True)

# ─────────────────────────────────────────────
# 성별 분석: 전체 성별 비율 + 권역×성별 스플릿
# ─────────────────────────────────────────────
with tab_gender:
    st.subheader("성별 분포(전체) 및 권역별 성별 비교")

    # 전체 성별 비율
    gender = df[df["성별_label"].notna()]["성별_label"]
    g_pct = (gender.value_counts(normalize=True) * 100).reindex(["남", "여"]).fillna(0).round(1)
    g_df = series_to_df(g_pct, "비율(%)", "성별")

    c1, c2 = st.columns([1, 2])
    with c1:
        pie = px.pie(
            g_df, values="비율(%)", names="성별", color="성별",
            color_discrete_map={"남": "#66c2a5", "여": "#fc8d62"},
            hole=0.4, title="성별 분포(%)"
        )
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
                color=alt.Color(
                    "성별_label:N", title="성별",
                    scale=alt.Scale(domain=["남", "여"], range=["#66c2a5", "#fc8d62"])
                ),
                tooltip=["권역:N", "성별_label:N", "비율(%):Q"]
            )
            .properties(title="권역별 성별 비율(권역 내 %)", height=320)
        )
        st.altair_chart(bar, use_container_width=True)

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

        # 시도명 컬럼 추정
        sido_col_candidates = [c for c in age_df.columns if "행정구역" in str(c)]
        sido_col = sido_col_candidates[0] if sido_col_candidates else age_df.columns[0]

        # 숫자형 연도 컬럼 중 최댓값(최신연도)
        year_cols = [c for c in age_df.columns if isinstance(c, (int, np.integer))]
        latest_year = max(year_cols)

        age_sido = age_df[[sido_col, latest_year]].rename(
            columns={sido_col: "시도", latest_year: "고령인구비율(%)"}
        )

        # 명칭 표준화
        age_sido["시도"] = age_sido["시도"].replace({
            "강원도": "강원특별자치도",
            "전라북도": "전북특별자치도"
        })

        # 권역 표준화 비율 재계산
        raw_counts = df["권역"].value_counts().reindex(VALID_REGIONS, fill_value=0)
        std_counts = raw_counts.astype(float).div(pd.Series(REGION_SIDO_N))
        std_pct = (std_counts / std_counts.sum() * 100)

        # 권역 → 시도 확장
        REGION_TO_SIDO = {
            "서울,인천": ["서울특별시", "인천광역시"],
            "경기,강원": ["경기도", "강원특별자치도"],
            "충청권(충북, 충남, 세종, 대전)": ["충청북도", "충청남도", "세종특별자치시", "대전광역시"],
            "전라권(전북, 전남, 광주)": ["전북특별자치도", "전라남도", "광주광역시"],
            "경상권(경북, 경남, 부산, 대구, 울산, 제주)":
                ["경상북도", "경상남도", "부산광역시", "대구광역시", "울산광역시", "제주특별자치도"],
        }

        expand_rows = []
        for reg, pct in std_pct.items():
            for s in REGION_TO_SIDO.get(reg, []):
                expand_rows.append({"시도": s, "표준화비율(%)": pct})
        reg_to_sido_df = pd.DataFrame(expand_rows)

        # 병합 & 시각화
        merged = pd.merge(reg_to_sido_df, age_sido, on="시도", how="inner")
        merged = merged.groupby("시도", as_index=False).mean(numeric_only=True)

        st.write("미리보기")
        st.dataframe(merged, use_container_width=True)

        # 상관계수
        corr = merged[["표준화비율(%)", "고령인구비율(%)"]].corr().iloc[0, 1]
        st.markdown(f"**상관계수 (권역 표준화비율 vs 시도 고령인구비율)**: `{corr:.2f}`")

        # 산점도 + 회귀선
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
st.caption(
    "ⓒ Respiratory Rehab / Pneumonia Insights — 권역은 요양기관 소재지 기준, "
    "권역 막대그래프는 시도수 보정(시도당 평균) 후 100% 정규화한 비율을 사용합니다."
)

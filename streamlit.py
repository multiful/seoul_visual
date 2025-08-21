# app.py — 기관종별 분포 + 권역 막대/지도 + 성별 분석
# - df_all.csv(호흡기 전체), pneumonia_data.csv(폐렴 전체)
# - 사이드바: 질환 대/중/상세 필터 + 폐렴 상세코드 multiselect
# - 환자데이터 권역 매핑 견고화(코드 A/B 자동감지 + 시도명 매핑 + 1~5 매핑)
# - 요양기관종별 코드 → 명칭 매핑(type_map)
# - '상위 개수' 슬라이더 제거(전체 표시)

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="환자 대시보드", page_icon="🫁")
alt.themes.enable("dark")
st.title("호흡기 질환 환자 대시보드")
st.caption("all_df=호흡기 전체, pneumonia_data=폐렴 전체 ")

# ─────────────────────────────────────────────
# 공용 유틸
# ─────────────────────────────────────────────
def series_to_df(s: pd.Series, value_name: str, index_name: str) -> pd.DataFrame:
    s = s.copy()
    df_tmp = s.to_frame(value_name)
    idx_name = index_name if index_name not in df_tmp.columns else f"{index_name}_idx"
    df_tmp = df_tmp.rename_axis(idx_name).reset_index()
    if idx_name != index_name:
        df_tmp = df_tmp.rename(columns={idx_name: index_name})
    return df_tmp

def map_sex(s):
    s = str(s).strip()
    if s in ("1", "남", "male", "Male", "M", "m"): return "남"
    if s in ("2", "여", "female", "Female", "F", "f"): return "여"
    return np.nan

def normalize_icd(code) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()

def first2digits(x: str) -> str:
    s = re.sub(r"\D", "", str(x))
    return s[:2] if len(s) >= 2 else s.zfill(2)

# 시도명 정규화(접미사 제거)
def norm_nm(s: str) -> str:
    s = re.sub(r"\s+", "", str(s))
    for t in ['특별자치도','특별자치시','특별시','광역시','자치도','도','시']:
        s = s.replace(t, '')
    return s

# ── 시도 코드체계(A/B) 매핑(지도/환자 공통)
CODE_TO_REGION_A = {
    "11": "서울,인천", "23": "서울,인천",
    "31": "경기,강원", "32": "경기,강원",
    "33": "충청권(충북, 충남, 세종, 대전)", "34": "충청권(충북, 충남, 세종, 대전)",
    "25": "충청권(충북, 충남, 세종, 대전)", "29": "충청권(충북, 충남, 세종, 대전)",
    "35": "전라권(전북, 전남, 광주)", "36": "전라권(전북, 전남, 광주)", "24": "전라권(전북, 전남, 광주)",
    "21": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "22": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    "26": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "37": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    "38": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "39": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
}
CODE_TO_REGION_B = {
    "11": "서울,인천", "28": "서울,인천",
    "41": "경기,강원", "42": "경기,강원",
    "43": "충청권(충북, 충남, 세종, 대전)", "44": "충청권(충북, 충남, 세종, 대전)",
    "36": "충청권(충북, 충남, 세종, 대전)", "30": "충청권(충북, 충남, 세종, 대전)",
    "45": "전라권(전북, 전남, 광주)", "46": "전라권(전북, 전남, 광주)", "29": "전라권(전북, 전남, 광주)",
    "47": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "48": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    "26": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "27": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    "31": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "50": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
}
NAME_TO_REGION = {
    "서울특별시": "서울,인천", "인천광역시": "서울,인천",
    "경기도": "경기,강원", "강원특별자치도": "경기,강원", "강원도": "경기,강원",
    "충청북도": "충청권(충북, 충남, 세종, 대전)", "충청남도": "충청권(충북, 충남, 세종, 대전)",
    "세종특별자치시": "충청권(충북, 충남, 세종, 대전)", "대전광역시": "충청권(충북, 충남, 세종, 대전)",
    "전북특별자치도": "전라권(전북, 전남, 광주)", "전라북도": "전라권(전북, 전남, 광주)",
    "전라남도": "전라권(전북, 전남, 광주)", "광주광역시": "전라권(전북, 전남, 광주)",
    "경상북도": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "경상남도": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    "부산광역시": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "대구광역시": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    "울산광역시": "경상권(경북, 경남, 부산, 대구, 울산, 제주)", "제주특별자치도": "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
}
# 정규화된 이름 → 권역(여분 백업용)
NORMNAME_TO_REGION = {
    '서울': "서울,인천", '인천': "서울,인천",
    '경기': "경기,강원", '강원': "경기,강원",
    '충북': "충청권(충북, 충남, 세종, 대전)", '충남': "충청권(충북, 충남, 세종, 대전)",
    '세종': "충청권(충북, 충남, 세종, 대전)", '대전': "충청권(충북, 충남, 세종, 대전)",
    '전북': "전라권(전북, 전남, 광주)", '전남': "전라권(전북, 전남, 광주)", '광주': "전라권(전북, 전남, 광주)",
    '경북': "경상권(경북, 경남, 부산, 대구, 울산, 제주)", '경남': "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    '부산': "경상권(경북, 경남, 부산, 대구, 울산, 제주)", '대구': "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
    '울산': "경상권(경북, 경남, 부산, 대구, 울산, 제주)", '제주': "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
}

REGION_MAP_1TO5 = {
    1: "서울,인천",
    2: "경기,강원",
    3: "충청권(충북, 충남, 세종, 대전)",
    4: "전라권(전북, 전남, 광주)",
    5: "경상권(경북, 경남, 부산, 대구, 울산, 제주)",
}
VALID_REGIONS = list(REGION_MAP_1TO5.values())
REGION_SIDO_N = {"서울,인천": 2, "경기,강원": 2, "충청권(충북, 충남, 세종, 대전)": 4, "전라권(전북, 전남, 광주)": 3, "경상권(경북, 경남, 부산, 대구, 울산, 제주)": 6}

def pick_region_mapping(codes_2digit: set[str]) -> dict:
    a_hits = len(set(CODE_TO_REGION_A) & codes_2digit)
    b_hits = len(set(CODE_TO_REGION_B) & codes_2digit)
    return CODE_TO_REGION_A if a_hits > b_hits else CODE_TO_REGION_B

def robust_region_from_records(df: pd.DataFrame) -> pd.Series:
    """
    환자 데이터에서 권역 라벨을 최대한 견고하게 생성.
      1) '요양기관소재지' 1~5 → REGION_MAP_1TO5
      2) 시도코드(두 자리) → A/B 코드 매핑
      3) 시도명(원래/정규화) → NAME_TO_REGION/NORMNAME_TO_REGION
    """
    # 1) 1~5
    if "요양기관소재지" in df.columns:
        s = pd.to_numeric(df["요양기관소재지"], errors="coerce")
        if s.notna().any():
            mapped = s.map(REGION_MAP_1TO5)
            if mapped.notna().mean() >= 0.8:
                return mapped

    # 2) 숫자코드(두 자리/다섯 자리) → A/B
    code_cols = [c for c in ["요양기관소재지", "시도코드", "CTPRVN_CD"] if c in df.columns]
    if code_cols:
        code_col = code_cols[0]
        two = df[code_col].astype(str).map(first2digits)
        mapping_used = pick_region_mapping(set(two.unique()))
        mapped = two.map(mapping_used)
        if mapped.notna().mean() >= 0.6:
            return mapped

    # 3) 시도명
    name_cols = [c for c in ["시도", "시도명", "CTP_KOR_NM", "광역시도", "요양기관광역"] if c in df.columns]
    if name_cols:
        nm = df[name_cols[0]].astype(str)
        m1 = nm.map(NAME_TO_REGION)
        if m1.notna().any():
            return m1
        m2 = nm.map(lambda x: NORMNAME_TO_REGION.get(norm_nm(x), np.nan))
        return m2

    return pd.Series([np.nan] * len(df), index=df.index, dtype="object")

@st.cache_resource(show_spinner=False)
def build_region_gdf(geo_path: str) -> tuple[gpd.GeoDataFrame, dict, dict]:
    gdf = gpd.read_file(geo_path)

    # CRS 처리
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

    # 지오메트리 유효화
    try:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.buffer(0)

    # 멀티폴리곤 분리
    try:
        gdf = gdf.explode(index_parts=False)
    except Exception:
        gdf = gdf.explode()

    # 코드 기반 매핑
    mapping_used = None
    gdf["권역_code"] = np.nan
    if "CTPRVN_CD" in gdf.columns:
        gdf["code2"] = gdf["CTPRVN_CD"].astype(str).map(first2digits)
        mapping_used = pick_region_mapping(set(gdf["code2"].unique()))
        gdf["권역_code"] = gdf["code2"].map(mapping_used)

    # 이름 기반 매핑(원본명 + 정규화명)
    gdf["권역_name"] = np.nan
    if "CTP_KOR_NM" in gdf.columns:
        nm = gdf["CTP_KOR_NM"].astype(str)
        name_map1 = nm.map(NAME_TO_REGION)
        name_map2 = nm.map(lambda x: NORMNAME_TO_REGION.get(norm_nm(x), np.nan))
        gdf["권역_name"] = name_map1.fillna(name_map2)

    # ★ 항상 코드 우선 + 이름 보정
    gdf["권역"] = gdf["권역_code"].astype("object")
    gdf["권역"] = gdf["권역"].where(gdf["권역"].notna(), gdf["권역_name"])

    # 커버리지 리포트
    unmapped = gdf[gdf["권역"].isna()][["CTPRVN_CD", "CTP_KOR_NM"]].drop_duplicates()
    coverage = {
        "총 시도수": int(gdf.drop_duplicates(subset=["CTPRVN_CD", "CTP_KOR_NM"]).shape[0]) if {"CTPRVN_CD","CTP_KOR_NM"}.issubset(gdf.columns) else int(gdf.shape[0]),
        "코드매핑_커버리지(%)": round(gdf["권역_code"].notna().mean() * 100, 1),
        "이름매핑_커버리지(%)": round(gdf["권역_name"].notna().mean() * 100, 1),
        "최종매핑_커버리지(%)": round(gdf["권역"].notna().mean() * 100, 1),
        "미매핑_시도": unmapped.to_dict(orient="records"),
        "사용한_코드매핑": ("A" if mapping_used is CODE_TO_REGION_A else "B") if mapping_used else "N/A",
    }

    # 특정 시도 면적 디버그
    dbg_names = ['충청남도','전라남도','부산광역시']
    dbg_area = {}
    for n in dbg_names:
        try:
            a = float(gdf.loc[gdf.get('CTP_KOR_NM','')==n, 'geometry'].area.sum())
            dbg_area[n] = a
        except Exception:
            dbg_area[n] = None
    coverage["디버그_면적m2"] = {k: (None if v is None else round(v,2)) for k,v in dbg_area.items()}

    # 권역 단위 dissolve
    region_gdf = gdf.dropna(subset=["권역"]).dissolve(by="권역", as_index=False)[["권역", "geometry"]]
    region_gdf = gpd.GeoDataFrame(region_gdf, geometry="geometry", crs=gdf.crs)

    # 최종 안전 보정
    try:
        region_gdf["geometry"] = region_gdf.buffer(0)
    except Exception:
        pass

    return region_gdf, {"mapping_used": mapping_used}, coverage

# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("데이터 선택")
    st.divider()

GEO_CANDIDATES = ["/mnt/data/TL_SCCO_CTPRVN.json", "TL_SCCO_CTPRVN.json", "data/TL_SCCO_CTPRVN.json"]
geo_path = next((p for p in GEO_CANDIDATES if Path(p).exists()), GEO_CANDIDATES[0])

ALL_FILE = "all_df.csv"             # 호흡기 전체 원천
PNEU_FILE = "pneumonia_data.csv"    # 폐렴 전체 원천

def read_csv_or_stop(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        st.error(f"'{path}' 파일을 찾을 수 없습니다.")
        st.stop()

df_all_raw  = read_csv_or_stop(ALL_FILE)
df_pneu_raw = read_csv_or_stop(PNEU_FILE)

# ─────────────────────────────────────────────
# 공통 라벨/매핑 상수
# ─────────────────────────────────────────────
TYPE_MAP = {10: "종합병원 이상", 21: "병원", 28: "요양병원", 29: "정신병원", 31: "의원", 41: "치과병원"}

# ─────────────────────────────────────────────
# 사이드바: 질환 필터(대/중/상세) + 폐렴 상세코드
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("세부분류 필터")

    # ICD 컬럼 자동 탐색(df_all 기준)
    main_candidates = ["주상병코드", "주상병", "주상병1", "주진단코드", "주진단"]
    sub_candidates  = ["부상병코드", "부상병", "부상병1", "부진단코드", "부진단"]
    single_candidates = ["상병코드", "진단코드", "ICD10", "IC-10", "질병코드", "주상병", "주상병코드", "상병"]

    def find_cols(df):
        main = next((c for c in main_candidates if c in df.columns), None)
        sub  = next((c for c in sub_candidates if c in df.columns), None)
        single = next((c for c in single_candidates if c in df.columns), None)
        return main, sub, single

    main_col_all, sub_col_all, single_col_all = find_cols(df_all_raw)

    def canon(s): return normalize_icd(s)

    # 대분류 매핑
    resp_disease_map = {
        "호흡기질환": ["J",'A15','A16','A19',"S270",'P251',"B664","B583","A430","A420","J690","J691","J698","J853"],
        "감기": ["J00","J01","J02","J03","J04","J05","J06"],
        "인플루엔자": ["J09", "J10", "J11"],
        "결핵": ["A15", "A16","A19"],
        "만성폐쇠질환(COPD)": ["J431", "J432","J438","J439","J40","J41","J42","J43","J44","J47"],
        "천식": ["J45","J46"],
        # 폐렴 루트/연관
        "폐렴": ["B664","B583","A430","A420","J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69","J85"],
        "기흉": ["J93", "S270", "P251"]
    }

    # 폐렴 상세코드 마스터
    pneumonia_codes_master = [
        "A420","A430","B583","B664",
        "J120","J121","J122","J128","J1280","J1288","J129",
        "J13","J14","J150","J151","J152","J153","J154","J155",
        "J156","J157","J158","J159","J160","J168","J170","J171",
        "J173","J178","J180","J181","J188","J189",
        "J690","J691","J698","J853"
    ]

    # 호흡기 전체(all_df) 구성
    def prepare_all_df(df_src: pd.DataFrame, main_col, sub_col, single_col) -> tuple[pd.DataFrame, str, str]:
        df2 = df_src.copy()
        if main_col is None and single_col is not None:
            main_col = single_col
        if sub_col is None:
            df2["_SUB_EMPTY_"] = ""
            sub_col = "_SUB_EMPTY_"
        for c in (main_col, sub_col):
            df2[c] = df2[c].apply(canon).astype("string")
        tb = tuple(resp_disease_map["결핵"])
        pneumo_extra = tuple([p for p in resp_disease_map["기흉"] if not p.startswith("J")])
        resp_mask = (
            df2[main_col].str.startswith("J", na=False) | df2[sub_col].str.startswith("J", na=False) |
            df2[main_col].str.startswith(tb, na=False) | df2[sub_col].str.startswith(tb, na=False) |
            df2[main_col].str.startswith(pneumo_extra, na=False) | df2[sub_col].str.startswith(pneumo_extra, na=False)
        )
        return df2[resp_mask].copy(), main_col, sub_col

    # 1) 대분류 선택
    super_labels = ["전체", "감기", "인플루엔자", "결핵", "만성폐쇠질환(COPD)", "천식", "폐렴", "기흉"]
    sel_super = st.selectbox("대분류 선택", super_labels, index=0, key="super_select")

    # 2) 데이터셋 선택 및 중/상세 필터
    if sel_super == "폐렴":
        df_base = df_pneu_raw.copy()
        m_p, s_p, one_p = find_cols(df_base)
        if m_p is None and one_p is not None:
            m_p = one_p
        if s_p is None:
            df_base["_SUB_EMPTY_"] = ""
            s_p = "_SUB_EMPTY_"
        for c in (m_p, s_p):
            df_base[c] = df_base[c].apply(canon).astype("string")

        pneu_roots = tuple(resp_disease_map["폐렴"])
        pool = pd.unique(pd.concat([df_base[m_p], df_base[s_p]], ignore_index=True)).astype(str)
        present_exact = {c for c in pool if c in set(pneumonia_codes_master)}
        present_prefix = {c for c in pool if any(c.startswith(pref) for pref in pneu_roots)}
        present = sorted(present_exact | present_prefix)

        sel_detail = st.multiselect(
            "폐렴 상세 코드 선택",
            options=["전체"] + present,
            default=["전체"],
            key="pneumonia_detail"
        )

        if "전체" in sel_detail or len(sel_detail) == 0:
            mask = (
                df_base[m_p].isin(pneumonia_codes_master) | df_base[s_p].isin(pneumonia_codes_master) |
                df_base[m_p].str.startswith(pneu_roots, na=False) |
                df_base[s_p].str.startswith(pneu_roots, na=False)
            )
        else:
            sels = tuple(sel_detail)
            mask = (
                df_base[m_p].isin(sel_detail) | df_base[s_p].isin(sel_detail) |
                df_base[m_p].str.startswith(sels, na=False) |
                df_base[s_p].str.startswith(sels, na=False)
            )

        df_selected = df_base[mask].copy()
        used_source = "폐렴 전체(pneumonia_data.csv)"
        chosen = "전체" if ("전체" in sel_detail or len(sel_detail) == 0) else ", ".join(sel_detail[:10]) + (" ..." if len(sel_detail) > 10 else "")
        st.caption(f"데이터 소스: pneumonia_data.csv (폐렴 전체) · 상세={chosen}")

    else:
        if (main_col_all is None) and (sub_col_all is None) and (single_col_all is None):
            st.error("df_all.csv에서 ICD 코드 컬럼을 찾지 못했습니다. (주/부상병 또는 상병코드류가 필요)")
            st.stop()
        all_df, main_used, sub_used = prepare_all_df(df_all_raw, main_col_all, sub_col_all, single_col_all)

        if sel_super == "전체":
            df_selected = all_df.copy()
            used_source = "호흡기질환 전체(df_all.csv)"
            st.caption("세부분류 필터 미적용 — 호흡기질환 전체(J*, 결핵 A15/A16/A19, 기흉 S270/P251 포함).")
        else:
            roots = resp_disease_map.get(sel_super, [])
            present = []
            for r in roots:
                m = all_df[main_used].str.startswith(r, na=False) | all_df[sub_used].str.startswith(r, na=False)
                if m.any():
                    present.append(r)
            if not present:
                st.info(f"선택한 대분류({sel_super})의 코드가 데이터에 없습니다. → 호흡기 전체로 대체")
                df_selected = all_df.copy()
                used_source = "호흡기질환 전체(df_all.csv)"
            else:
                sel_mid = st.selectbox("중분류(루트) 선택", options=["전체"] + present, index=0, key=f"mid_{sel_super}")
                if sel_mid == "전체":
                    mask = False
                    for r in present:
                        mask = mask | all_df[main_used].str.startswith(r, na=False) | all_df[sub_used].str.startswith(r, na=False)
                    df_selected = all_df[mask].copy()
                    used_source = "호흡기질환 전체(df_all.csv)"
                    st.caption(f"적용된 세부분류: {sel_super} / 전체 루트({len(present)}개)")
                else:
                    codes_main = all_df.loc[all_df[main_used].str.startswith(sel_mid, na=False), main_used]
                    codes_sub  = all_df.loc[all_df[sub_used].str.startswith(sel_mid, na=False), sub_used]
                    subs_present = sorted(pd.Index(codes_main.tolist() + codes_sub.tolist()).unique())
                    sel_detail = st.multiselect(
                        f"상세 코드 선택 ({sel_mid}*)",
                        options=["전체"] + subs_present,
                        default=["전체"],
                        key=f"detail_{sel_mid}"
                    )
                    if "전체" in sel_detail or len(sel_detail) == 0:
                        mask = all_df[main_used].str.startswith(sel_mid, na=False) | all_df[sub_used].str.startswith(sel_mid, na=False)
                    else:
                        sels = tuple(sel_detail)
                        mask = (
                            all_df[main_used].isin(sel_detail) | all_df[sub_used].isin(sel_detail) |
                            all_df[main_used].str.startswith(sels, na=False) | all_df[sub_used].str.startswith(sels, na=False)
                        )
                    df_selected = all_df[mask].copy()
                    used_source = "호흡기질환 전체(df_all.csv)"
                    chosen = "전체" if ("전체" in sel_detail or len(sel_detail) == 0) else ", ".join(sel_detail[:10]) + (" ..." if len(sel_detail) > 10 else "")
                    st.caption(f"적용된 세부분류: {sel_super} / {sel_mid} / {chosen}")

    # 이후 단계에서 쓸 공용 df
    df = df_selected.copy()
    st.caption(f"현재 레코드 수(연령 필터 전): {len(df):,}")

# ─────────────────────────────────────────────
# 연령대 필터(선택 df 위에 적용)
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("연령대 필터")
    age_col = next((c for c in ["연령", "나이"] if c in df.columns), None)
    if age_col is None or df[age_col].dropna().empty:
        st.info("연령/나이 데이터가 없어 연령대 필터를 표시하지 않습니다.")
    else:
        age_numeric = pd.to_numeric(df[age_col], errors="coerce")
        max_age_val = np.nanmax(age_numeric.values)
        max_bin = int(np.ceil(max(max_age_val, 0) / 10.0) * 10) if np.isfinite(max_age_val) else 10
        max_bin = max(10, max_bin)
        bins = list(range(0, max_bin + 10, 10))
        labels_age = [f"{b}대" for b in bins[:-1]]
        df["연령대"] = pd.cut(age_numeric, bins=bins, right=False, labels=labels_age)
        df["연령대"] = df["연령대"].astype("string").fillna("미상")

        selected_bands = []
        for lab in labels_age:
            if st.toggle(lab, value=True, key=f"ageband_{lab}"):
                selected_bands.append(lab)
        if selected_bands:
            df = df[df["연령대"].isin(selected_bands)]
        st.caption("선택된 연령대: " + (", ".join(selected_bands) if selected_bands else "모두(미상 제외)"))
    st.caption(f"현재 레코드 수(연령 필터 적용 후): {len(df):,}")

# ─────────────────────────────────────────────
# 공통 전처리(선택 df에 대해)
# ─────────────────────────────────────────────
# 권역 라벨(견고 매핑)
df["권역"] = robust_region_from_records(df)

# 요양기관종별 코드 → 명칭
if "요양기관종별" in df.columns:
    type_code_num = pd.to_numeric(df["요양기관종별"], errors="coerce")
    df["요양기관종별_명칭"] = type_code_num.map(TYPE_MAP).astype("string")
elif "요양기관종별_명칭" in df.columns:
    df["요양기관종별_명칭"] = df["요양기관종별_명칭"].astype("string")
else:
    df["요양기관종별_명칭"] = pd.Series(["미상"] * len(df), dtype="string")

# 성별 라벨
df["성별_label"] = df.get("성별", pd.Series(index=df.index)).map(map_sex)

# 분석 대상만 남기기(권역 유효)
df = df[df["권역"].isin(VALID_REGIONS)].copy()

# ─────────────────────────────────────────────
# 메인: 요양기관종별 '분포' (전체 표시)
# ─────────────────────────────────────────────
st.subheader("요양기관종별 분포")
type_col = "요양기관종별_명칭"
type_series = df[type_col].dropna().astype(str)

if type_series.empty:
    st.info("요양기관종별 데이터가 없습니다.")
else:
    counts = type_series.value_counts()
    pct = (counts / counts.sum() * 100).round(2)
    cnt_df = series_to_df(counts, "건수", type_col)
    pct_df = series_to_df(pct, "비율(%)", type_col)
    type_df = cnt_df.merge(pct_df, on=type_col).sort_values("건수", ascending=False)

    metric = st.radio("표시 기준", ["건수", "비율(%)"], horizontal=True)
    show_col = "건수" if metric == "건수" else "비율(%)"
    chart_df = type_df.sort_values(show_col, ascending=False)

    cT1, cT2 = st.columns([2, 1], gap="large")
    with cT1:
        bar = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{show_col}:Q"),
                y=alt.Y(f"{type_col}:N", sort="-x", title="요양기관종별"),
                tooltip=[type_col, "건수", "비율(%)"],
                color=alt.Color(f"{show_col}:Q", scale=alt.Scale(scheme="reds")),
            )
            .properties(height=max(280, 22 * len(chart_df)))
        )
        text = (
            alt.Chart(chart_df)
            .mark_text(align="left", baseline="middle", dx=4)
            .encode(x=f"{show_col}:Q", y=f"{type_col}:N", text=f"{show_col}:Q")
        )
        st.altair_chart(bar + text, use_container_width=True)
    with cT2:
        donut = px.pie(chart_df, values=show_col, names=type_col, hole=0.5, title="요양기관 비중(요약)")
        donut.update_traces(textinfo="percent+label")
        st.plotly_chart(donut, use_container_width=True)
    with st.expander("표(요양기관종별 분포)"):
        st.dataframe(type_df, use_container_width=True)

# ─────────────────────────────────────────────
# 권역: 시도수 (표준화) 막대 + Choropleth 지도
# ─────────────────────────────────────────────
st.subheader("권역별 분포 — 시도수(표준화) 기준")

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
            color=alt.Color("시도수 보정 비율(%):Q", scale=alt.Scale(scheme="reds")),
        )
        .properties(height=360)
    )
    text = (
        alt.Chart(plot_df)
        .mark_text(align="left", baseline="middle", dx=4)
        .encode(x="시도수 보정 비율(%):Q", y="권역:N", text="시도수 보정 비율(%):Q")
    )
    st.altair_chart(chart + text, use_container_width=True)

with c2:
    try:
        region_gdf, debug_map, coverage = build_region_gdf(geo_path)
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
        fig_map.update_layout(height=420, margin=dict(l=0, r=0, t=60, b=0),
                              coloraxis_colorbar=dict(title="시도수 보정<br>비율(%)"),
                                  title="권역별 시도수 보정 비율(%) 지도", title_y=0.95)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"지도를 생성하는 데 문제가 발생했습니다: {e}")

with st.expander("표(권역 시도수 보정 비율)"):
    st.dataframe(plot_df, use_container_width=True)

# ─────────────────────────────────────────────
# 성별 분석 (파이 + 권역×성별 막대 + 표)
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
    with st.expander("표(성별 전체 비율)"):
        st.dataframe(g_df, use_container_width=True)

with c4:
    if len(df[df["성별_label"].notna()]) > 0:
        cross_counts = (
            df[df["성별_label"].notna()]
            .groupby(["권역", "성별_label"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        cross_counts["비율(%)"] = cross_counts.groupby("권역")["count"].transform(lambda s: s / s.sum() * 100)
        cross = cross_counts[["권역", "성별_label", "비율(%)"]]

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
        with st.expander("표(권역×성별 비율)"):
            st.dataframe(cross.sort_values(["권역", "성별_label"]), use_container_width=True)
    else:
        st.info("성별 정보가 없어 권역×성별 막대를 표시할 수 없습니다.")

# ─────────────────────────────────────────────
# 풋노트
# ─────────────────────────────────────────────
st.caption("ⓒ Respiratory Rehab / Pneumonia Insights — 권역은 요양기관 소재지 기준, 권역 막대그래프는 시도수 보정(시도당 평균) 후 100% 정규화한 비율을 사용합니다.")


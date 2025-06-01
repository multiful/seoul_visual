# 📦 패키지 임포트
import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# 📌 페이지 설정
st.set_page_config(page_title='서울시 복지시설 시각화 대시보드', layout='wide')

# 📍 사이드바 설정
st.sidebar.title("필터 설정")
year = st.sidebar.selectbox("연도 선택", options=[2018, 2019, 2020, 2021, 2022], index=2)
columns_to_plot = [
    '가정폭력피해자보호시설',
    '성매매피해자보호시설',
    '소계_소외여성복지시설',
    '미혼모자_공동생활가정',
    '미혼모자시설',
    '양육시설',
    '여성_부랑인_시설',
    '합계_아동복지시설'
]
selected_columns = st.sidebar.multiselect("시각화할 지표 선택", options=columns_to_plot, default=columns_to_plot[:4])

# 📁 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_excel('Result/데이터셋 통합.xlsx')
    df['연도'] = df['연도'].fillna(method='ffill')
    df.set_index(['연도', '자치구'], inplace=True)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    return df

@st.cache_data
def load_geo():
    return gpd.read_file('seoul_municipalities_geo_simple.json', encoding='utf-8')

df = load_data()
geo_df = load_geo()

# 📊 연도 필터링
df_year = df.xs(year, level='연도').reset_index()
geo_df = geo_df.merge(df_year, left_on='SIG_KOR_NM', right_on='자치구')

# 📌 대시보드 제목
st.title(f"서울시 복지시설 분포 지도 ({year}년)")

# 📊 지도 시각화
fig, axes = plt.subplots(nrows=1, ncols=len(selected_columns), figsize=(6 * len(selected_columns), 8))
if len(selected_columns) == 1:
    axes = [axes]

for idx, col in enumerate(selected_columns):
    ax = axes[idx]
    geo_df.plot(
        column=col,
        cmap='OrRd',
        linewidth=0.5,
        edgecolor='black',
        ax=ax,
        legend=True,
        legend_kwds={'shrink': 0.6}
    )
    
    for _, row in geo_df.iterrows():
        centroid = row['geometry'].centroid
        gu_name = row['SIG_KOR_NM']
        txt = ax.text(
            centroid.x, centroid.y, gu_name,
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=0.5, foreground='black'),
            pe.Normal()
        ])
    
    ax.set_title(f'{col}', fontsize=12)
    ax.set_axis_off()

plt.tight_layout()
st.pyplot(fig)

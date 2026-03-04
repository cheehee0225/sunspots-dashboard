import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

st.set_page_config(layout="wide", page_title="태양흑점 데이터 분석 대시보드")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # 안전하게 숫자 변환
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["SUNACTIVITY"] = pd.to_numeric(df["SUNACTIVITY"], errors="coerce")
    df = df.dropna(subset=["YEAR", "SUNACTIVITY"]).copy()

    # YEAR 정수화
    df["YEAR_INT"] = df["YEAR"].astype(int)

    # datetime index 생성
    df["DATE"] = pd.to_datetime(df["YEAR_INT"].astype(str), format="%Y")
    df = df.set_index("DATE").sort_index()

    return df


def plot_advanced_sunspot_visualizations(
    df,
    sunactivity_col="SUNACTIVITY",
    hist_bins=30,
    trend_degree=1,
    point_size=10,
    point_alpha=0.5,
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Sunspots Data Advanced Visualization", fontsize=18)

    # (a) 전체 시계열 라인 차트
    axs[0, 0].plot(df.index, df[sunactivity_col], color="blue")
    axs[0, 0].set_title("Sunspot Activity Over Time")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_ylabel("Sunspot Count")
    axs[0, 0].grid(True)

    # (b) 분포: 히스토그램 + 커널 밀도
    data = df[sunactivity_col].dropna().to_numpy()

    if len(data) > 0:
        # ✅ 여기가 너 코드에서 들여쓰기 깨져서 에러 난 부분!
        axs[0, 1].hist(
            data,
            bins=hist_bins,
            density=True,
            alpha=0.6,
            color="gray",
            label="Histogram",
        )

        # KDE는 데이터가 너무 적거나(1개) 모두 같은 값이면 터질 수 있음 → 방어
        if len(data) > 1 and np.std(data) > 0:
            xs = np.linspace(data.min(), data.max(), 200)
            try:
                density = gaussian_kde(data)
                axs[0, 1].plot(xs, density(xs), color="red", linewidth=2, label="Density")
            except Exception:
                pass

    axs[0, 1].set_title("Distribution of Sunspot Activity")
    axs[0, 1].set_xlabel("Sunspot Count")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # (c) 상자 그림: 1900년~2000년
    try:
        df_20th = df.loc["1900":"2000"]
        if not df_20th.empty:  # ✅ 콜론(:) 반드시!
            axs[1, 0].boxplot(df_20th[sunactivity_col].dropna(), vert=False)
    except:
        pass

    axs[1, 0].set_title("Boxplot of Sunspot Activity (1900-2000)")
    axs[1, 0].set_xlabel("Sunspot Count")

    # (d) 산점도 + 회귀선
    years = df["YEAR_INT"].to_numpy(dtype=float)
    sun_activity = df[sunactivity_col].to_numpy(dtype=float)

    mask = ~np.isnan(sun_activity)
    years_clean = years[mask]
    sun_activity_clean = sun_activity[mask]

    if len(years_clean) > 0:
        axs[1, 1].scatter(
            years_clean,
            sun_activity_clean,
            s=point_size,
            alpha=point_alpha,
            label="Data Points",
        )

    # polyfit은 degree+1개 이상의 점이 필요
    if len(years_clean) >= trend_degree + 1:
        coef = np.polyfit(years_clean, sun_activity_clean, trend_degree)
        trend = np.poly1d(coef)
        x_trend = np.linspace(years_clean.min(), years_clean.max(), 100)
        axs[1, 1].plot(x_trend, trend(x_trend), color="red", linewidth=2, label="Trend Line")

    axs[1, 1].set_title("Trend of Sunspot Activity")
    axs[1, 1].set_xlabel("Year")
    axs[1, 1].set_ylabel("Sunspot Count")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# -----------------------
# 메인 앱
# -----------------------
st.title("🌞 태양흑점 데이터 분석 대시보드 🌞")
st.markdown("이 대시보드는 태양흑점 데이터를 다양한 시각화 방법으로 보여줍니다.")

try:
    # ✅ 배포/로컬 모두에서 안전한 경로
    DATA_PATH = Path(__file__).parent / "data" / "sunspots.csv"
    df = load_data(DATA_PATH)

    # 사이드바 슬라이더
    st.sidebar.header("⚙️ 대시보드 설정")

    min_year = int(df["YEAR_INT"].min())
    max_year = int(df["YEAR_INT"].max())

    year_range = st.sidebar.slider(
        "연도 범위",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )

    hist_bins = st.sidebar.slider("히스토그램 구간 수", 5, 100, 30)
    trend_degree = st.sidebar.slider("추세선 차수", 1, 5, 1)
    point_size = st.sidebar.slider("산점도 점 크기", 1, 100, 10)
    point_alpha = st.sidebar.slider("산점도 투명도", 0.05, 1.0, 0.5, step=0.05)

    # 필터링
    filtered_df = df[(df["YEAR_INT"] >= year_range[0]) & (df["YEAR_INT"] <= year_range[1])]

    if filtered_df.empty:
        st.warning("선택한 기간에 데이터가 없습니다.")
    else:
        st.subheader("태양흑점 데이터 종합 시각화")
        fig = plot_advanced_sunspot_visualizations(
            filtered_df,
            hist_bins=hist_bins,
            trend_degree=trend_degree,
            point_size=point_size,
            point_alpha=point_alpha,
        )
        st.pyplot(fig)

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
    st.info("data/sunspots.csv 파일이 존재하고 'YEAR', 'SUNACTIVITY' 컬럼이 있어야 합니다.")

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 설정
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 군집화)")

# 2. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    total_raw_count = len(df)

    # 파생 속성 및 결측치/조건 처리
    # 필수 열에 결측치가 있거나 first_week_audi가 0인 행 제거
    df = df.dropna(
        subset=[
            "first_scrn",
            "total_audi",
            "days_in_top10",
            "first_week_audi",
            "movieNm",
        ]
    )
    df = df[df["first_week_audi"] > 0].copy()

    # 파생 속성 생성
    # 롱런 지수 = 누적 관객 / 첫 주 관객 (최대 20)
    df["long_run_idx"] = (df["total_audi"] / df["first_week_audi"]).clip(
        upper=20
    )

    # 상용로그 변환 (스크린 수, 누적 관객)
    # log10 적용 시 0 또는 음수 방지를 위해 np.log10 사용
    df["log_first_scrn"] = np.log10(df["first_scrn"].replace(0, np.nan))
    df["log_total_audi"] = np.log10(df["total_audi"].replace(0, np.nan))

    df = df.dropna(
        subset=["log_first_scrn", "log_total_audi", "long_run_idx"]
    )
    filtered_count = len(df)

    return df, total_raw_count, filtered_count


df, raw_count, clean_count = load_data()

# 데이터 편수 안내
st.write(
    f"**전체 영화 편수:** {raw_count:,}편 | **분석 대상 편수(조건 정제 후):** {clean_count:,}편"
)

st.divider()

# 3. 군집화 속성 선택
feature_dict = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 수 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_idx",
}

st.subheader("1. 군집화에 사용할 속성 선택")
selected_feature_names = st.multiselect(
    "군집화에 활용할 속성을 선택하세요 (2개 이상 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys()),
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 군집화를 위해 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_dict[name] for name in selected_feature_names]

# 4. K-Means 군집화 수행
# 데이터 표준화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_features])

# K-Means (3개 그룹, random_state 고정)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)
df["raw_cluster"] = cluster_labels

# 그룹명 재정의 (누적 관객 평균(total_audi)이 높은 순서대로 ㉮, ㉯, ㉰ 부여)
avg_audi_per_cluster = df.groupby("raw_cluster")["total_audi"].mean()
sorted_clusters = avg_audi_per_cluster.sort_values(ascending=False).index

label_map = {
    sorted_clusters[0]: "㉮",
    sorted_clusters[1]: "㉯",
    sorted_clusters[2]: "㉰",
}

df["cluster"] = df["raw_cluster"].map(label_map)
cluster_order = ["㉮", "㉯", "㉰"]

# 5. 시각화 섹션
st.divider()
st.subheader("2. 군집 시각화")

# 2차원 산점도
col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "2차원 X축 속성 선택:",
        options=list(feature_dict.keys()),
        index=0,
        key="2d_x",
    )
with col2:
    y_axis_2d = st.selectbox(
        "2차원 Y축 속성 선택:",
        options=list(feature_dict.keys()),
        index=1,
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=feature_dict[x_axis_2d],
    y=feature_dict[y_axis_2d],
    color="cluster",
    category_orders={"cluster": cluster_order},
    hover_name="movieNm",
    labels={
        feature_dict[x_axis_2d]: x_axis_2d,
        feature_dict[y_axis_2d]: y_axis_2d,
        "cluster": "유형",
    },
    title=f"2차원 산점도 ({x_axis_2d} vs {y_axis_2d})",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.write("---")
if len(selected_feature_names) >= 3:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox(
            "3차원 X축 속성 선택:",
            options=list(feature_dict.keys()),
            index=0,
            key="3d_x",
        )
    with col4:
        y_axis_3d = st.selectbox(
            "3차원 Y축 속성 선택:",
            options=list(feature_dict.keys()),
            index=1,
            key="3d_y",
        )
    with col5:
        z_axis_3d = st.selectbox(
            "3차원 Z축 속성 선택:",
            options=list(feature_dict.keys()),
            index=2,
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_axis_3d],
        y=feature_dict[y_axis_3d],
        z=feature_dict[z_axis_3d],
        color="cluster",
        category_orders={"cluster": cluster_order},
        hover_name="movieNm",
        labels={
            feature_dict[x_axis_3d]: x_axis_3d,
            feature_dict[y_axis_3d]: y_axis_3d,
            feature_dict[z_axis_3d]: z_axis_3d,
            "cluster": "유형",
        },
        title=f"3차원 산점도 ({x_axis_3d} vs {y_axis_3d} vs {z_axis_3d})",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    # 점 크기 축소
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("ℹ️ 선택한 속성이 3개 미만입니다. 3차원 산점도를 보려면 속성을 3개 이상 선택해 주세요.")

# 6. 통계 및 대표 영화 통계
st.divider()
st.subheader("3. 묶음(유형)별 통계 및 대표 영화")

# 평균 통계 표 계산 (원래 단위 표현)
summary_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객수_평균=("total_audi", "mean"),
        상위10위_일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_idx", "mean"),
    )
    .reindex(cluster_order)
)

# 표 출력 포맷 정의
st.dataframe(
    summary_df.style.format(
        {
            "편수": "{:,}편",
            "스크린수_평균": "{:,.1f}개",
            "누적관객수_평균": "{:,.0f}명",
            "상위10위_일수_평균": "{:,.1f}일",
            "롱런지수_평균": "{:,.2f}",
        }
    ),
    use_container_width=True,
)

# 묶음별 누적 관객수 상위 5개 영화 목록
st.write("#### 🏆 묶음별 누적 관객수 Top 5 영화")
cols = st.columns(3)

for idx, cluster_name in enumerate(cluster_order):
    with cols[idx]:
        st.markdown(f"### 묶음 {cluster_name}")
        top5_movies = (
            df[df["cluster"] == cluster_name]
            .sort_values(by="total_audi", ascending=False)
            .head(5)
        )
        for i, (_, row) in enumerate(top5_movies.iterrows(), 1):
            st.write(f"**{i}. {row['movieNm']}** ({row['total_audi']:,}명)")

"""Model Card Studio のエントリポイント。UI は薄く、計算は src/ に委譲する。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.data import available_datasets, load_dataset
from src.limitations import generate_limitations
from src.metrics import (
    apply_threshold,
    compute_calibration,
    compute_metrics,
    confusion_counts,
    threshold_recall_swing,
)
from src.model import predict_proba_positive, train_model
from src.plots import plot_calibration, plot_confusion_matrix

DATASET_LABELS = {
    "breast_cancer": "Breast Cancer（sklearn・比較的バランス）",
    "imbalanced_synthetic": "不均衡合成データ（95:5）",
}


@st.cache_data(show_spinner=False)
def train_pipeline(dataset_name: str, test_size: float, random_state: int) -> dict:
    """データ読込→学習→テスト予測確率までをキャッシュする。しきい値は含めない。"""
    ds = load_dataset(dataset_name, test_size=test_size, random_state=random_state)
    model = train_model(ds.X_train, ds.y_train, random_state=random_state)
    y_prob = predict_proba_positive(model, ds.X_test)
    return {
        "y_test": ds.y_test,
        "y_prob": y_prob,
        "pos_label": ds.pos_label,
        "neg_label": ds.neg_label,
        "feature_names": ds.feature_names,
        "description": ds.description,
        "source": ds.source,
        "pos_count": ds.pos_count,
        "neg_count": ds.neg_count,
        "total_count": ds.total_count,
        "n_train": ds.X_train.shape[0],
        "n_test": ds.X_test.shape[0],
    }


def main() -> None:
    st.set_page_config(
        page_title="Model Card Studio",
        page_icon="📋",
        layout="centered",
    )
    from src.brand import apply_brand, hero
    apply_brand(st)
    hero(st, "Model Reporting", "Model Card Studio", "分類モデルの評価をモデルカード形式で表示し、校正・混同行列も可視化します。")
    st.caption("分類モデルの評価を、精度以外も含めてモデルカード形式で見せるアプリ")

    with st.sidebar:
        st.header("設定")
        dataset_name = st.selectbox(
            "データセット",
            options=list(available_datasets()),
            format_func=lambda x: DATASET_LABELS.get(x, x),
            index=0,
        )
        test_size = st.slider(
            "テストデータ割合", min_value=0.2, max_value=0.5, value=0.3, step=0.05
        )
        random_state = st.number_input(
            "乱数シード", min_value=0, max_value=9999, value=42, step=1
        )
        threshold = st.slider(
            "判定しきい値（陽性確率）", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        st.divider()
        if st.button("再学習（キャッシュクリア）", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_retrain"] = True
            st.rerun()
        if st.session_state.get("last_retrain"):
            st.caption("キャッシュをクリアして再学習しました。")

    try:
        bundle = train_pipeline(dataset_name, test_size, int(random_state))
    except Exception as exc:  # noqa: BLE001
        st.error(f"学習に失敗しました: {exc}")
        return

    if bundle["n_test"] == 0:
        st.warning("テストデータが空です。テストデータ割合を上げてください。")
        return

    y_test = bundle["y_test"]
    y_prob = bundle["y_prob"]
    y_pred = apply_threshold(y_prob, threshold)
    metrics = compute_metrics(y_test, y_pred)
    confusion = confusion_counts(y_test, y_pred)
    calibration = compute_calibration(y_test, y_prob, n_bins=10)
    recall_swing = threshold_recall_swing(y_test, y_prob, threshold)
    limitations = generate_limitations(
        metrics,
        confusion,
        calibration,
        bundle,
        threshold,
        recall_swing,
    )

    tab_summary, tab_metrics, tab_cm, tab_cal, tab_lim = st.tabs(
        ["データセット概要", "評価指標", "混同行列", "較正", "制限事項"]
    )

    with tab_summary:
        st.subheader("データセット概要")
        st.write(bundle["description"])
        c1, c2, c3 = st.columns(3)
        c1.metric("総サンプル数", bundle["total_count"])
        c2.metric("学習 / テスト", f"{bundle['n_train']} / {bundle['n_test']}")
        c3.metric("特徴量数", len(bundle["feature_names"]))
        c1, c2 = st.columns(2)
        c1.metric(bundle["pos_label"], bundle["pos_count"])
        c2.metric(bundle["neg_label"], bundle["neg_count"])
        ratio = (
            max(bundle["pos_count"], bundle["neg_count"])
            / max(1, min(bundle["pos_count"], bundle["neg_count"]))
        )
        st.info(f"クラス比（多数:少数）= {ratio:.1f}:1")
        st.caption(f"データ出所: {bundle['source']}")

    with tab_metrics:
        st.subheader("評価指標")
        st.caption(f"しきい値 = {threshold:.2f} のとき")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("正解率", f"{metrics['accuracy']:.3f}")
        c2.metric("適合率", f"{metrics['precision']:.3f}")
        c3.metric("再現率", f"{metrics['recall']:.3f}")
        c4.metric("F1", f"{metrics['f1']:.3f}")
        c1, c2 = st.columns(2)
        c1.metric("評価サンプル数", metrics["support"])
        c2.metric("しきい値±0.1 の再現率変動", f"{recall_swing:.3f}")
        if abs(metrics["accuracy"] - metrics["recall"]) >= 0.1:
            st.warning(
                f"正解率 {metrics['accuracy']:.2f} と再現率 {metrics['recall']:.2f} に差がある。"
                "精度だけではモデルを評価できない例。"
            )

    with tab_cm:
        st.subheader("混同行列")
        fig = plot_confusion_matrix(confusion, bundle["pos_label"], bundle["neg_label"])
        st.pyplot(fig)
        st.caption(
            f"TP={confusion['tp']} / FP={confusion['fp']} / "
            f"TN={confusion['tn']} / FN={confusion['fn']}"
        )

    with tab_cal:
        st.subheader("較正")
        c1, c2 = st.columns(2)
        c1.metric("Brierスコア", f"{calibration['brier']:.3f}")
        c2.metric("ECE", f"{calibration['ece']:.3f}")
        fig = plot_calibration(calibration)
        st.pyplot(fig)
        st.caption("点の大きさは bin 内サンプル数に概ね比例。対角線に近いほど較正が良い。")

    with tab_lim:
        st.subheader("制限事項（自動生成）")
        st.markdown(
            "**精度だけではモデルを評価できない。** 以下は指標とデータ特性から自動抽出した注意点。"
        )
        for i, item in enumerate(limitations, 1):
            st.markdown(f"{i}. {item}")

    st.divider()
    st.caption(
        "Model Card Studio ・ sklearn / 合成データのみ使用・APIキー不要・"
        "計算ロジックは src/・UI は app/streamlit_app.py"
    )


if __name__ == "__main__":
    main()

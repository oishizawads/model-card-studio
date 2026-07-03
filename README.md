# Model Card Studio

## 目的

分類モデルの評価結果を、非技術者にも説明しやすい**モデルカード**形式で表示する小型アプリ。
正解率（Accuracy）だけを見せて終わるのではなく、適合率・再現率・混同行列・較正・制限事項まで一套で見せることで、
**精度だけではモデルを評価できない**ことを実感できるようにする。

## 主要機能

- 二値分類モデル（ロジスティック回帰）を学習し、テストデータで評価する
- 評価指標：正解率 / 適合率 / 再現率 / F1（しきい値を動かすと連動して変化）
- 混同行列の可視化（TP/FP/TN/FN を明示）
- 較正（キャリブレーション）の可視化：信頼性曲線・Brierスコア・ECE
- 制限事項の**自動文章化**：不均衡・低再現率・較正悪化・しきい値感度・小サンプル等を指標から検出
- データセット切替：バランス型（Breast Cancer）と意図的不均衡型（95:5 合成データ）

## 使用技術

- Python 3.11+
- Streamlit（UI）
- scikit-learn（データセット・学習）
- matplotlib（混同行列・較正曲線）
- pandas / numpy
- pytest（テスト）

## データの出所

外部データは使わない。すべて scikit-learn 付属のサンプルデータまたは合成データを用いる。

| データセット | 出所 | 特徴 |
|---|---|---|
| Breast Cancer | `sklearn.datasets.load_breast_cancer` | 比較的バランスの取れた二値分類 |
| 不均衡合成データ | `sklearn.datasets.make_classification`（`weights=[0.95, 0.05]`） | 意図的に 95:5 のクラス不均衡 |

APIキー・外部サービス・ネットワーク接続は不要。

## ローカル実行手順

```bash
cd model-card-studio

# 仮想環境と依存関係を整える（uv を推奨）
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# アプリ起動
streamlit run app/streamlit_app.py

# テスト実行
pytest
```

> uv を使わない場合: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

起動後、サイドバーでデータセット・テスト割合・乱数シードを変え、**判定しきい値スライダー**を動かすと、
評価指標タブの数値と混同行列が連動して更新される。

## スクリーンショット

画面キャプチャは `assets/` に配置する（例: `assets/dashboard.png`）。

## 制限事項

- MVP 範囲：認証・DB・本番運用・課金・複雑な状態管理は持たない
- モデルはロジスティック回帰のみ。他アルゴリズムの比較は今後の課題
- 較正の補正（Platt / isotonic）は可視化までで、自動適用はしない
- データは sklearn サンプル/合成データであり、実運用の母集団を代表しない

## 精度だけではモデルを評価できない

正解率が高くても、クラス不均衡下では少数クラスの再現率が低くなり、実運用で重大な取りこぼしを生む。
本アプリでは、しきい値を動かしたときの適合率と再現率の乖離、較正曲線のズレ、自動生成された制限事項を通じて、
単一指標（とくに正解率）だけでの評価が危険であることを示す。

## ディレクトリ構成

```
model-card-studio/
├── app/streamlit_app.py   # エントリポイント（UI は薄く）
├── src/                   # 計算ロジック
│   ├── data.py            # データ読込
│   ├── model.py           # 学習・予測確率
│   ├── metrics.py         # 指標・混同行列・較正
│   ├── limitations.py     # 制限事項の自動文章化
│   └── plots.py           # matplotlib 描画
├── tests/                 # pytest（src/ 評価関数）
├── assets/                # スクリーンショット置き場
└── pyproject.toml
```

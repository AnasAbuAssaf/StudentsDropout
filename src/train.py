# ══════════════════════════════════════════════════════════════
# 📌 الخطوة 7.1 — الملف src/train.py (تدريب وحفظ)
# Pipeline واحد (imputer + scaler + one-hot + نموذج) يُقارَن بخط أساس ويُحفظ مع بصمة البيانات (Lineage).
# ══════════════════════════════════════════════════════════════
"""Stage: train — fit ONE Pipeline, compare with a baseline, save model + data hash."""
import json

import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from common import *

# ميزات لحظة التنبؤ
features = features_at(PREDICTION_POINT)
# جدول point-in-time
pit = pd.read_parquet(DATA / "features_pit.parquet")
# التسميات مع مصدرها
labels = pd.read_csv(DATA / "labels_v1.csv")
# التقسيم الثابت
split = pd.read_csv(DATA / "split_ids.csv")
# دمج الميزات والتسميات والتقسيم
data = pit.merge(labels[["student_id", TARGET, "label_source"]], on="student_id").merge(split[["student_id", "split"]], on="student_id")
# التدريب والاختبار
tr, te = data[data.split == "train"], data[data.split == "test"]
# الهدف كـ 0/1
y_tr, y_te = to_binary(tr[TARGET]), to_binary(te[TARGET])

# بوابة P5: نسبة الفقد يجب ألا تتجاوز عتبة العقد
# P5 gate: missing rate must respect the contract threshold
# أعلى نسبة فقد بين الأعمدة
miss = float(tr[features].isna().mean().max())
# البوابة: فقد فوق 5% يوقف التدريب
gate("p5_missing_rate_within_contract", miss <= MAX_MISSING_RATE, f"max column missing rate {miss:.2%} (limit {MAX_MISSING_RATE:.0%})")

# خط الأساس: يتنبأ دائماً بالفئة الأكثر شيوعاً (prior). أي نموذج يجب أن يتفوق عليه
baseline = DummyClassifier(strategy="prior").fit(tr[features], y_tr)
# تدريب الـ Pipeline الواحد، وكل إحصاءاته تُتعلَّم هنا مرة واحدة
model = make_pipeline(features).fit(tr[features], y_tr)          # every statistic is learned here, once
# احتمال الانسحاب على الاختبار
proba = model.predict_proba(te[features])[:, 1]
# المقاييس: macro-F1 للنموذج وخط الأساس، الدقة، ROC-AUC، وأحجام المجموعتين
metrics = {"macro_f1": float(f1_score(y_te, model.predict(te[features]), average="macro")),
           "baseline_macro_f1": float(f1_score(y_te, baseline.predict(te[features]), average="macro")),
           "accuracy": float(accuracy_score(y_te, model.predict(te[features]))),
           "roc_auc": float(roc_auc_score(y_te, proba)), "n_train": int(len(tr)), "n_test": int(len(te))}
# حفظ المقاييس
(REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
# البوابة: النموذج أسوأ من خط الأساس بالهامش المطلوب ← يتوقف الخط (شريحة 26)
gate("model_beats_baseline", metrics["macro_f1"] > metrics["baseline_macro_f1"] + 0.05,
     f"macro-F1 {metrics['macro_f1']:.3f} vs baseline {metrics['baseline_macro_f1']:.3f}")

# بصمة md5 لكل مدخل: البيانات النظيفة، التقسيم، التسميات، الميزات
inputs = {p.name: md5_file(p) for p in [DATA / "clean.parquet", DATA / "split_ids.csv", DATA / "labels_v1.csv", DATA / "features_pit.parquet"]}
# حفظ الـ Pipeline كاملاً. هذا الملف هو الذي تحمّله الخدمة (Rule #32)
joblib.dump(model, MODELS / "model.joblib")                       # the serving job loads THIS file
# بيانات النموذج: نقطة التنبؤ، الميزات، إصدار sklearn، بصمات المدخلات، بصمة موحّدة data_hash، المقاييس
meta = {"prediction_point": PREDICTION_POINT, "features": features, "sklearn": sklearn.__version__,
        "inputs_md5": inputs, "data_hash": hashlib.md5("".join(inputs.values()).encode()).hexdigest(),
        "metrics": metrics, "label_source": sorted(data["label_source"].unique().tolist())}
# حفظ البيانات الوصفية لتتبع النسب
(MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
# طباعة بصمة البيانات
print("saved models/model.joblib | data_hash:", meta["data_hash"])

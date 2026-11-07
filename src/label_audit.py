# ══════════════════════════════════════════════════════════════
# 📌 الخطوة 4.1 — الملف src/label_audit.py (تدقيق التسميات P4)
# (1) اتفاق المرمّزين بمقياس Cohen κ. (2) كشف الأخطاء المحتملة بـ confident learning.
# (3) جدول تسميات موثَّق المصدر. المراجعة بشرية ولا حذف تلقائي.
# ══════════════════════════════════════════════════════════════
"""Stage: label_audit — kappa on a double-labelled sample + cleanlab review queue (P4)."""
import json

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from common import *

# عتبة κ تُحدَّد مسبقاً (0.60 فأكثر = «substantial» عند Landis & Koch)
KAPPA_THRESHOLD = 0.60     # fixed in advance (Landis & Koch: >= 0.6 'substantial')
# محاكاة: المرمّز الثاني يقلب 8% من التسميات
ANNOTATOR2_NOISE = 0.08    # SIMULATED second annotator: flips 8% of labels
# حجم العيّنة مزدوجة الترميز
SAMPLE = 300
# أقصى نسبة صفوف مشتبه بها قبل أن نتوقف للتحقيق (20%)
LABEL_ISSUE_LIMIT = 0.20   # share of train rows flagged above which we stop and investigate

# دمج البيانات النظيفة مع عمود التقسيم
df = pd.read_parquet(DATA / "clean.parquet").merge(pd.read_csv(DATA / "split_ids.csv")[["student_id", "split"]], on="student_id")
# نعمل على جزء التدريب فقط حتى لا يتسرب شيء من الاختبار
train = df[df.split == "train"].reset_index(drop=True)
# الهدف كـ 0/1
y = to_binary(train[TARGET]).to_numpy()

# 1) اتفاق المرمّزين: κ = (pₒ − pₑ) ÷ (1 − pₑ) حيث pₒ الاتفاق الملاحَظ وpₑ المتوقع بالصدفة
# 1) inter-annotator agreement on a double-labelled sample
# مولد عشوائي بذرته ثابتة
rng = np.random.default_rng(SEED)
# اختيار عيّنة عشوائية بلا إرجاع
idx = rng.choice(len(train), size=min(SAMPLE, len(train)), replace=False)
# المرمّز الأول = التسمية الفعلية
a1 = y[idx]
# المرمّز الثاني (محاكى): يقلب كل تسمية باحتمال 8%
a2 = np.where(rng.random(len(idx)) < ANNOTATOR2_NOISE, 1 - a1, a1)
# حساب κ بين المرمّزين
kappa = float(cohen_kappa_score(a1, a2))
# نسبة الاتفاق الخام (للمقارنة مع κ)
raw_agree = float((a1 == a2).mean())
# حفظ تقرير κ مع وسم أن المرمّز الثاني محاكى
(REPORTS / "kappa.json").write_text(json.dumps({"kappa": kappa, "raw_agreement": raw_agree, "n": int(len(idx)),
                                                "threshold": KAPPA_THRESHOLD, "annotator2": "SIMULATED"}, indent=2))
# البوابة: κ أقل من العتبة ← إصلاح دليل الترميز وإيقاف الخط
gate("p4_kappa_above_threshold", kappa >= KAPPA_THRESHOLD, f"kappa={kappa:.3f} (raw agreement {raw_agree:.1%}), threshold {KAPPA_THRESHOLD}")

# 2) Confident learning: نقدّر احتمالات خارج العيّنة (cross-validated) لأن الاحتمالات داخل العيّنة حفظت الضجيج
# 2) confident learning on OUT-OF-SAMPLE probabilities (features available at t0 only)
# ميزات لحظة التنبؤ فقط
feats = features_at(PREDICTION_POINT)
# تقسيم تحقق متقاطع طبقي بحسب المجموعة، فلا تتسرب النسخ بين الطيّات
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
# احتمالات كل صف من نموذج لم يرَ هذا الصف
probs = cross_val_predict(make_pipeline(feats), train[feats], y, cv=cv, groups=train["group_id"], method="predict_proba")
# فهارس الصفوف المشتبه في تسميتها، مرتبة من الأقل ثقة بتسميتها
issues = find_label_issues(labels=y, pred_probs=probs, return_indices_ranked_by="self_confidence")
# أسماء الفئات للعرض
name = {0: "Graduate", 1: "Dropout"}
# طابور المراجعة label_issues.csv: التسمية المعطاة والمقترحة ودرجة الثقة
pd.DataFrame({"student_id": train["student_id"].iloc[issues].to_numpy(),
              "given_label": [name[int(v)] for v in y[issues]],
              "suggested_label": [name[int(v)] for v in probs[issues].argmax(1)],
              "self_confidence": probs[issues, y[issues]]}).to_csv(REPORTS / "label_issues.csv", index=False)
# نسبة الصفوف المشتبه بها من التدريب
share = len(issues) / len(train)

# 3) جدول التسميات مع المصدر: الصفوف المشتبه بها تُراجَع ولا تُحذف بصمت
# 3) labels table with provenance — flagged rows are REVIEWED, never silently deleted
# مجموعة معرّفات الطلاب المشتبه بهم
flagged = set(train["student_id"].iloc[issues])
# عمود label_source (مصدر التسمية) وإصدار الدليل وعلامة وجودها في طابور المراجعة
labels = df[["student_id", TARGET]].assign(label_source="registrar_outcome", guideline_version="v1",
                                           in_review_queue=df["student_id"].isin(flagged).astype(int))
# حفظ labels_v1.csv
labels.to_csv(DATA / "labels_v1.csv", index=False)
# البوابة: نسبة المشتبه بها فوق الحد تعني مشكلة في التسميات أو النموذج فنتوقف
gate("p4_flagged_share_below_limit", share <= LABEL_ISSUE_LIMIT, f"{len(issues)} flagged = {share:.1%} of train (limit {LABEL_ISSUE_LIMIT:.0%})")

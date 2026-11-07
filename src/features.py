# ══════════════════════════════════════════════════════════════
# 📌 الخطوة 6.4 — الملف src/features.py (جدول point-in-time)
# لا طوابع زمنية في البيانات، فنحاكي تقويماً أكاديمياً (SIMULATED):
#   التسجيل سبتمبر 2020 | الفصل 1 بعد 150 يوماً + تأخر تصحيح | الفصل 2 بعد 330 يوماً + تأخر.
#   t₀ = التسجيل + 180 يوماً، و δ = 7 أيام هامش أمان.
# الهدف (شريحة 22): لكل طالب نأخذ آخر قيمة بطابع ≤ t₀ − δ، ولا نستخدم المستقبل.
# ══════════════════════════════════════════════════════════════
"""Stage: features — timestamped feature tables + POINT-IN-TIME training table (P6).

The dataset has NO timestamps. SIMULATED academic calendar (one cohort):
  enrolment values   known at enrolment date (Sept 2020, +0..13 days per student)
  semester-1 values  known 150 days later + 5..20 days of grading lag
  semester-2 values  known 330 days later + 5..20 days of grading lag
  prediction time t0 = enrolment + 180 days (start of 2nd semester); delta = 7 days ingestion safety margin.
"""
import numpy as np
import pandas as pd

from common import *

# δ: التأخر بين وجود القيمة وإتاحتها للخدمة (7 أيام)
SERVING_DELTA = pd.Timedelta(days=7)
# تاريخ بدء التسجيل (محاكى)
ENROLL_BASE = pd.Timestamp("2020-09-14", tz="UTC")
# عروض الميزات وأعمدتها
VIEWS = {"enroll_features": ENROLL_FEATURES, "sem1_features": SEM1, "sem2_features": SEM2}
# العروض المستخدمة عند لحظة التنبؤ المختارة (sem1 يستخدم التسجيل والفصل الأول فقط)
USED = {"enroll": ["enroll_features"], "sem1": ["enroll_features", "sem1_features"], "sem2": list(VIEWS)}[PREDICTION_POINT]
# نوع الطابع الزمني بالتوقيت العالمي لضمان تطابق الأنواع
NS = "datetime64[ns, UTC]"

# البيانات النظيفة
df = pd.read_parquet(DATA / "clean.parquet")
# عدد الطلاب
n = len(df)
# مولد عشوائي لتواريخ التسجيل وتأخر التصحيح
rng = np.random.default_rng(SEED)
# تاريخ تسجيل كل طالب: بين 0 و13 يوماً بعد التاريخ الأساسي
enroll_ts = pd.Series(ENROLL_BASE + pd.to_timedelta(rng.integers(0, 14, n), unit="D")).astype(NS)
# وقت إتاحة كل مجموعة ميزات: تسجيل | فصل 1 (150 + 5..20 يوم) | فصل 2 (330 + 5..20 يوم)
ts = {"enroll_features": enroll_ts,
      "sem1_features": (enroll_ts + pd.to_timedelta(150 + rng.integers(5, 21, n), unit="D")).astype(NS),
      "sem2_features": (enroll_ts + pd.to_timedelta(330 + rng.integers(5, 21, n), unit="D")).astype(NS)}
# لحظة التنبؤ t₀ = التسجيل + 180 يوماً
t0 = (enroll_ts + pd.Timedelta(days=180)).astype(NS)

# جداول الميزات المؤرَّخة = المخزن offline (وهي نفسها مصدر ملفات Feast)
# timestamped feature tables = the OFFLINE store (also the source files for Feast)
# قاموس الجداول
views = {}
# لكل عرض: نبني جدولاً مؤرَّخاً
for name, cols in VIEWS.items():
    # المفتاح + أعمدة العرض
    v = df[["student_id"] + cols].copy()
    # تحويل النوع: العشري إلى float32 والباقي إلى int64 (يطابق مخطط Feast)
    for c in cols:
        # تطبيق النوع
        v[c] = v[c].astype("float32" if c in FLOAT_COLS else "int64")
    # إضافة وقت إتاحة القيمة
    v["event_timestamp"] = ts[name].to_numpy()
    v["event_timestamp"] = v["event_timestamp"].astype(NS)
    # حفظ الجدول في feature_repo/data ليقرأه Feast
    v.to_parquet(ROOT / "feature_repo" / "data" / f"{name}.parquet", index=False)
    # حفظ الجدول لاستخدامه في الربط
    views[name] = v

# الربط الزمني الصحيح: آخر قيمة بطابع ≤ t₀ − δ
# POINT-IN-TIME join: latest value with timestamp <= t0 - delta
# جدول الكيانات: لكل طالب لحظة تنبؤه t₀
entity = pd.DataFrame({"student_id": df["student_id"].to_numpy(), "t0": t0.to_numpy()})
# توحيد نوع الطابع
entity["t0"] = entity["t0"].astype(NS)
# حد القطع = t₀ − δ. لا يجوز استخدام أي قيمة بطابع بعده
entity["cutoff"] = (entity["t0"] - SERVING_DELTA).astype(NS)
# merge_asof يتطلب ترتيباً زمنياً
pit = entity.sort_values("cutoff").reset_index(drop=True)
# نضم كل عرض مستخدم بالتسلسل
for name in USED:
    # تسمية عمود الطابع باسم العرض لنميّز الطوابع
    v = views[name].rename(columns={"event_timestamp": f"{name}_ts"}).sort_values(f"{name}_ts")
    # ربط تراجعي (backward): لكل طالب آخر قيمة بطابع ≤ cutoff، مع التقييد بـ student_id
    pit = pd.merge_asof(pit, v, left_on="cutoff", right_on=f"{name}_ts", by="student_id", direction="backward")

# أعمدة الطوابع للعروض المستخدمة
ts_cols = [f"{name}_ts" for name in USED]
# ميزات لحظة التنبؤ
features = features_at(PREDICTION_POINT)
# ترتيب الأعمدة: المفتاح، t₀، القطع، الطوابع، ثم الميزات، وترتيب الصفوف بحسب student_id
pit = pit[["student_id", "t0", "cutoff"] + ts_cols + features].sort_values("student_id").reset_index(drop=True)
# حفظ جدول التدريب point-in-time
pit.to_parquet(DATA / "features_pit.parquet", index=False)

# نسبة قيم الفصل 2 التي هي مستقبل عند t₀. ربط «آخر قيمة» العادي كان سيسرّبها
future_share = float((ts["sem2_features"] > (t0 - SERVING_DELTA)).mean())
# عدد القيم التي طابعها بعد حد القطع (يجب أن يكون 0)
violations = int(sum((pit[c] > pit["cutoff"]).sum() for c in ts_cols))
# نسبة الطلاب الذين لهم كل الميزات المستخدمة
coverage = float(pit[ts_cols].notna().all(axis=1).mean())
# عرض نسبة التسريب المحتمل لو استخدمنا ربطاً ساذجاً
print(f"semester-2 values that are in the FUTURE at t0: {future_share:.0%} (a 'latest value' join would leak them)")
# البوابة 1: لا طابع زمني بعد t₀ − δ
gate("p6_no_feature_timestamp_after_t0_minus_delta", violations == 0, f"{violations} violations")
# البوابة 2: تغطية كاملة لكل الطلاب
gate("p6_point_in_time_full_coverage", coverage == 1.0, f"{coverage:.1%} of students have every used feature")
# ميزات الفصل 2 الداخلة في مدخلات النموذج رغم أنها مستقبل (يجب ألا توجد)
leak = [c for c in features if c in SEM2 and PREDICTION_POINT != "sem2"]
# البوابة 3: لا ميزة مستقبلية في مدخلات النموذج
gate("p6_no_future_feature_in_model_inputs", not leak, f"{len(features)} features at t0 = {PREDICTION_POINT}")

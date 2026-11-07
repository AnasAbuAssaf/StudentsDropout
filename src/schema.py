# ══════════════════════════════════════════════════════════════
# 📌 الخطوة 2.1 — الملف src/schema.py (عقد البيانات بـ Pandera)
# العقد = Schema + عتبات الجودة + الحداثة + المالك (شريحة 9).
# مستويات الفحص: schema (أنواع) | value (مدى) | row (علاقة بين حقول) | dataset (عدد الصفوف).
# ══════════════════════════════════════════════════════════════
"""Data contract for the student-retention table (P2) — schema + thresholds + owner."""
import json

import pandera.pandas as pa
from pandera import Check

from common import *

# عدد الصفوف المرجعي من الملف المرجعي (4424 افتراضياً إن لم يوجد بعد)
REF_ROWS = (json.loads((REPORTS / "reference_profile.json").read_text())["rows"]
            if (REPORTS / "reference_profile.json").exists() else 4424)
# بيانات العقد: الاسم والمالك (من يُنبَّه عند الفشل) والمرجع وهامش الصفوف ونسبة الفقد القصوى والحداثة
CONTRACT = {"name": "student_retention_v1", "owner": "registrar-data@example.edu",
            "reference_rows": REF_ROWS, "row_tolerance": 0.20, "max_null_rate": 0.0,
            "freshness": "re-validate every new batch"}

# المدى المسموح لكل عمود رقمي: (النوع، الحد الأدنى، الحد الأقصى). حدود واسعة لكنها ذات معنى
RANGES = {   # column: (dtype, min, max) — generous but meaningful bounds
    "application_order": (int, 0, 20), "previous_qualification_grade": (float, 0, 200),
    "admission_grade": (float, 0, 200), "age_at_enrollment": (int, 16, 100),
    "unemployment_rate": (float, 0, 30), "inflation_rate": (float, -5, 15), "gdp": (float, -10, 10),
}


# يبني عموداً بنوع ومدى محدَّدين
def _range(dtype, lo, hi):
    # التحويل القسري يجعل القيمة النصية مثل 1,5 خطأً صريحاً بدل أن تصير NaN بصمت، ويرفض NaN بـ nullable=False
    # coerce=True turns "1,5" into an ERROR (not a silent NaN); NaN is rejected by nullable=False
    return pa.Column(dtype, Check.in_range(lo, hi), nullable=False, coerce=True)


# عمود student_id: عدد صحيح غير سالب، فريد، بلا فقد
columns = {"student_id": pa.Column(int, Check.ge(0), unique=True, nullable=False, coerce=True)}
# الأعمدة الرقمية ذات المدى
columns.update({c: _range(*RANGES[c]) for c in RANGES})
# الأعمدة الفئوية: أعداد صحيحة غير سالبة
columns.update({c: pa.Column(int, Check.ge(0), nullable=False, coerce=True) for c in ENROLL_CAT})
# الأعمدة الثنائية: القيمة 0 أو 1 فقط
columns.update({c: pa.Column(int, Check.isin([0, 1]), nullable=False, coerce=True) for c in ENROLL_BIN})
# أعمدة الفصلين: الدرجات عشرية بين 0 و20 والباقي أعداد صحيحة بين 0 و100
for c in SEM1 + SEM2:
    columns[c] = _range(float, 0, 20) if c.endswith("_grade") else _range(int, 0, 100)
# عمود الهدف: يجب أن يكون من الفئات المسموحة وبلا فقد
columns[TARGET] = pa.Column(checks=Check.isin(CLASSES_ALL), nullable=False)

# هامش تغيّر عدد الصفوف (20%)
_tol = CONTRACT["row_tolerance"]
# تجميع كل الأعمدة والفحوص في عقد واحد
schema = pa.DataFrameSchema(
    columns,
    checks=[
        # فحص مستوى المجموعة: عدد الصفوف ضمن ±20% من الدفعة المرجعية، وإلا فالدفعة مشبوهة
        # dataset level
        Check(lambda d: (1 - _tol) * REF_ROWS <= len(d) <= (1 + _tol) * REF_ROWS,
              name="row_count_within_20pct", error="row count drifted > 20% from the reference batch"),
        # فحص مستوى الصف (ناعم): المنجَز ≤ المسجَّل. ينتج تحذيراً ولا يوقف الخط؛ تحقق منه على نسختك قبل جعله صارماً
        # row level (soft: warn, do not stop — verify on your copy before making it a hard check)
        Check(lambda d: d["curricular_units_1st_sem_approved"] <= d["curricular_units_1st_sem_enrolled"],
              name="sem1_approved_le_enrolled", raise_warning=True),
    ],
    # أي عمود إضافي يُعدّ خطأً (حقل جديد من المورّد يجب أن يُراجَع)
    strict=True,            # extra columns = error (a new vendor field must be reviewed)
    name=CONTRACT["name"],
)

# ══════════════════════════════════════════════════════════════
# 📌 الخطوة 2.2 — الملف src/validate.py (مرحلة التحقق)
# تفحص كل دفعة بالعقد؛ عند الفشل: عزل الدفعة وإيقاف الخط (شريحة 9).
# ══════════════════════════════════════════════════════════════
"""Stage: validate  — contract check on every batch; fail => quarantine + stop."""
import json
import warnings

import pandas as pd
import pandera.pandas as pa

from common import *
from schema import CONTRACT, schema

# قراءة الدفعة الجديدة
df = load_raw()
# نلتقط التحذيرات (الفحص الناعم) لنعدّها
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    # lazy=True يجمع كل الإخفاقات بدل التوقف عند أولها. النجاح يعطي valid والفشل يعطي جدول failures
    try:
        valid, failures = schema.validate(df, lazy=True), pd.DataFrame()   # lazy: report ALL failures
    # عند الفشل نأخذ جدول حالات الفشل failure_cases
    except pa.errors.SchemaErrors as err:
        valid, failures = None, err.failure_cases

# عدد التحذيرات الناعمة
soft = sum("sem1_approved_le_enrolled" in str(w.message) for w in caught)
# ملخص الإخفاقات
summary = {}
# تجميع الإخفاقات حسب (العمود، الفحص) مع العدد
if len(failures):
    summary = (failures.assign(column=failures["column"].fillna("<dataframe>"))
               .groupby(["column", "check"]).size().rename("n").reset_index().to_dict("records"))
# تقرير التحقق: العقد، عدد الصفوف، هل نجح، عدد الإخفاقات، التحذيرات
report = {"contract": CONTRACT, "rows": int(len(df)), "passed": valid is not None,
          "n_failure_cases": int(len(failures)), "soft_warnings": int(soft), "failures": summary}
# حفظ تقرير التحقق
(REPORTS / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))

# عند الفشل: نحفظ حالات الفشل ونعزل الدفعة في data/quarantine
if valid is None:   # quarantine the batch and alert the owner
    # تفاصيل كل إخفاق (الصف، العمود، القيمة)
    failures.to_csv(REPORTS / "validation_failures.csv", index=False)
    # مجلد العزل
    (DATA / "quarantine").mkdir(exist_ok=True)
    # نسخة الدفعة المعزولة
    df.to_csv(DATA / "quarantine" / "batch.csv", index=False)
# البوابة: أي فشل في العقد يوقف الخط (والمالك يُنبَّه)
gate("p2_data_contract", valid is not None,
     f"{len(df)} rows, {len(failures)} failing cases, {soft} soft warning(s); owner={CONTRACT['owner']}")
# عند النجاح فقط: حفظ البيانات الصالحة لتغذي المرحلة التالية
valid.to_parquet(DATA / "valid.parquet", index=False)

# ══════════════════════════════════════════════════════════════
# 📌 الخطوة 3.1 — الملف src/split.py (النطاق + حذف التكرار + تقسيم بحسب المجموعة)
# الهدف (P3): ألا يكون لمثال الاختبار نسخة أو شبه نسخة في التدريب، وإلا قسنا الحفظ لا التعميم.
# ══════════════════════════════════════════════════════════════
"""Stage: split — scope, exact dedupe, near-duplicate groups, group split (P1 + P3)."""
import json

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import *

# نصف قطر شبه التكرار في فضاء الميزات المعيَّرة (ε). الصفان الأقرب منه يُعدّان شبه نسختين
NEAR_DUP_EPS = 1.0     # radius in standardised feature space
# نسبة الاختبار 20%
TEST_SIZE = 0.20

# قراءة البيانات الصالحة من مرحلة التحقق
df = pd.read_parquet(DATA / "valid.parquet")
# عدد الصفوف الصالحة قبل أي حذف
n_valid = len(df)

# P1: استبعاد الفئة المُراقَبة Enrolled
# P1 scope: drop the censored class
df = df[df[TARGET].isin(SCOPE_CLASSES)].reset_index(drop=True)
# عدد الصفوف داخل النطاق
n_scope = len(df)

# P3.1: النسخ المطابقة تُحذف قبل التقسيم. keep=False يعلّم كل النسخ لنعدّها
# P3.1 exact duplicates (everything except the surrogate key) — BEFORE the split
# قناع الصفوف المكررة في الميزات (دون student_id لأنه اصطناعي)
dup_mask = df.duplicated(subset=ALL_FEATURES, keep=False)
# عدد مجموعات النسخ التي لها تسميات متعارضة (مؤشر ضجيج تسميات)
conflicting = int(df[dup_mask].groupby(ALL_FEATURES)[TARGET].nunique().gt(1).sum()) if dup_mask.any() else 0
# الإبقاء على نسخة واحدة من كل صف مطابق في الميزات والهدف
df = df.drop_duplicates(subset=ALL_FEATURES + [TARGET]).reset_index(drop=True)
# عدد النسخ المطابقة المحذوفة
n_exact = n_scope - len(df)

# P3.2: شبه التكرار. نبني رسماً: بين كل صفين أقرب من ε حافة، والمكوّنات المتصلة = مجموعات
# P3.2 near-duplicates: connected components of the "closer than eps" graph
# توحيد قياس الميزات ليسهم كل منها بالتساوي في المسافة
Z = StandardScaler().fit_transform(df[ALL_FEATURES].astype(float))
# رسم الجيران ضمن نصف القطر (مصفوفة اتصال متفرقة)
graph = NearestNeighbors(radius=NEAR_DUP_EPS).fit(Z).radius_neighbors_graph(Z, mode="connectivity")
# المكوّنات المتصلة: كل مكوّن = مجموعة نسخ قريبة. تؤدي دور leaf_id في مثال PlantVillage
n_groups, group_id = connected_components(graph, directed=False)
# إضافة رقم المجموعة لكل صف
df["group_id"] = group_id
# حجم مجموعة كل صف (أكبر من 1 = له شبه نسخة)
sizes = df.groupby("group_id")["student_id"].transform("size")

# P3.3: التقسيم بحسب المجموعة، فالمجموعة الواحدة تذهب كلها إلى التدريب أو كلها إلى الاختبار
# P3.3 one group -> one side
# مقسِّم المجموعات بنسبة اختبار 20% وبذرة ثابتة
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
# فهارس التدريب والاختبار
tr_idx, te_idx = next(gss.split(df, groups=df["group_id"]))
# الافتراضي تدريب ثم نعلّم صفوف الاختبار
df["split"] = "train"
# تعليم صفوف الاختبار
df.loc[te_idx, "split"] = "test"

# المجموعات الظاهرة في الجهتين معاً. يجب أن تكون فارغة
both = set(df.loc[df.split == "train", "group_id"]) & set(df.loc[df.split == "test", "group_id"])
# تقرير التقسيم: أعداد الصفوف والنسخ والمجموعات ونسبة الانسحاب في كل جهة
report = {"rows_valid": n_valid, "rows_in_scope": n_scope, "exact_duplicates_removed": int(n_exact),
          "conflicting_label_duplicates": conflicting, "near_dup_eps": NEAR_DUP_EPS,
          "groups": int(n_groups), "rows_in_multi_row_groups": int((sizes > 1).sum()),
          "train": int((df.split == "train").sum()), "test": int((df.split == "test").sum()),
          "dropout_share_train": float((df[df.split == "train"][TARGET] == POSITIVE).mean()),
          "dropout_share_test": float((df[df.split == "test"][TARGET] == POSITIVE).mean()),
          "groups_in_both_sets": len(both)}
# حفظ التقرير
(REPORTS / "split_report.json").write_text(json.dumps(report, indent=2))

# حفظ البيانات النظيفة (دون عمود split)
df.drop(columns="split").to_parquet(DATA / "clean.parquet", index=False)
# حفظ التقسيم الثابت split_ids.csv (يُصدَّر مع البيانات ويبقى كما هو في كل إعادة تدريب)
df[["student_id", "group_id", "split"]].to_csv(DATA / "split_ids.csv", index=False)   # fixed + versioned
# البوابة: وجود أي مجموعة في التدريب والاختبار معاً يوقف الخط
gate("p3_no_group_in_both_sets", len(both) == 0,
     f"{len(both)} shared groups; {n_exact} exact dups removed; {report['rows_in_multi_row_groups']} rows in near-dup groups")

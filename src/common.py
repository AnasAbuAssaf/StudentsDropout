# ══════════════════════════════════════════════════════════════
# 📌 الخطوة 0.3 — الملف src/common.py (الأساس المشترك)
# يضم: الثوابت، مجموعات الأعمدة، دوال التحميل، بوابة gate()،
# تعريف الميزة الواحدة، وبناء الـ Pipeline الواحد (Rule #32).
# الأمر %%writefile يكتب محتوى هذه الخلية في ملف ولا ينفّذه.
# ══════════════════════════════════════════════════════════════
"""Shared helpers for the W02 student-retention data pipeline."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# جذر المشروع = المجلد الأب لمجلد src؛ منه نبني كل المسارات
ROOT = Path(__file__).resolve().parents[1]
# مسارات المجلدات: البيانات، التقارير، النماذج
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"
# مسار ملف البيانات الخام
RAW_CSV ="/kaggle/input/datasets/anasaboassaf/dropout"
# إنشاء المجلدات إن لم تكن موجودة (آمن عند التكرار)
for _d in (DATA / "raw", REPORTS, MODELS, ROOT / "feature_repo" / "data"):
    _d.mkdir(parents=True, exist_ok=True)

# بذرة عشوائية ثابتة لإعادة إنتاج النتائج نفسها في كل تشغيل
SEED = 42
# اسم عمود الهدف، والفئة الموجبة = الانسحاب (Dropout)
TARGET, POSITIVE = "target", "Dropout"
# كل فئات الهدف المسموحة (يتحقق منها عقد البيانات في P2)
CLASSES_ALL = ["Dropout", "Enrolled", "Graduate"]
# نطاق المشكلة (P1): نستبعد Enrolled لأن طلابها بلا نتيجة نهائية بعد (بيانات مُراقَبة censored) فلا تصلح كتسميات
SCOPE_CLASSES = ["Dropout", "Graduate"]   # P1 scope: 'Enrolled' has no final outcome yet (censored)
# لحظة التنبؤ t₀: enroll = عند التسجيل | sem1 = نهاية الفصل الأول | sem2 = نهاية الفصل الثاني
PREDICTION_POINT = "sem1"                 # t0 = end of 1st semester: "enroll" | "sem1" | "sem2"
# عتبة العقد: أقصى نسبة فقد مسموحة في أي عمود (تستخدمها بوابة P5)
MAX_MISSING_RATE = 0.05                   # contract threshold used by the P5 gate

# ---- column groups (names AFTER normalisation) -------------------------------------
# ميزات رقمية تُعرف عند التسجيل
ENROLL_NUM = ["application_order", "previous_qualification_grade", "admission_grade", "age_at_enrollment"]
# ميزات فئوية (أرقام ترميزية للمقرر، الوالدين...) ستُرمَّز One-Hot داخل الـ Pipeline
ENROLL_CAT = ["marital_status", "application_mode", "course", "previous_qualification", "nationality",
              "mother_s_qualification", "father_s_qualification", "mother_s_occupation", "father_s_occupation"]
# ميزات ثنائية 0/1 تُعامل كرقمية
ENROLL_BIN = ["daytime_evening_attendance", "displaced", "educational_special_needs", "debtor",
              "tuition_fees_up_to_date", "gender", "scholarship_holder", "international"]
# مؤشرات اقتصاد كلي (بطالة، تضخم، ناتج محلي) من تغذية خارجية سنوية
MACRO = ["unemployment_rate", "inflation_rate", "gdp"]
# إحصاءات الوحدات الدراسية التي تُسجَّل لكل فصل
_STATS = ["credited", "enrolled", "evaluations", "approved", "grade", "without_evaluations"]
# أعمدة الفصلين الأول والثاني تُولَّد أسماؤها آلياً من _STATS. الفصل الأول متاح عند t₀=sem1 والثاني مستقبل
SEM1 = [f"curricular_units_1st_sem_{s}" for s in _STATS]
SEM2 = [f"curricular_units_2nd_sem_{s}" for s in _STATS]
# الأعمدة العشرية (الباقي أعداد صحيحة)، تُستخدم لتحديد النوع في Pandera وFeast
FLOAT_COLS = ["previous_qualification_grade", "admission_grade", "unemployment_rate", "inflation_rate", "gdp",
              "curricular_units_1st_sem_grade", "curricular_units_2nd_sem_grade"]
# ميزات لحظة التسجيل = رقمية + فئوية + ثنائية + اقتصادية
ENROLL_FEATURES = ENROLL_NUM + ENROLL_CAT + ENROLL_BIN + MACRO
# كل الميزات: تسجيل + فصل 1 + فصل 2
ALL_FEATURES = ENROLL_FEATURES + SEM1 + SEM2
# الأعمدة المتوقعة في الملف الخام (يفحصها load_raw كعقد مصغّر)
EXPECTED_COLUMNS = ALL_FEATURES + [TARGET]


# تُرجع الميزات المتاحة فقط عند لحظة التنبؤ المحددة، فلا يدخل النموذج أي معلومة من المستقبل (منع التسريب)
def features_at(point=PREDICTION_POINT):
    """Features that exist at prediction time t0."""
    return {"enroll": ENROLL_FEATURES, "sem1": ENROLL_FEATURES + SEM1, "sem2": ALL_FEATURES}[point]


# ---- loading -----------------------------------------------------------------------
# توحيد أسماء الأعمدة: أحرف صغيرة وشرطة سفلية بدل المسافات والرموز
def normalize_columns(df):
    # نسخة حتى لا نعدّل الجدول الأصلي
    out = df.copy()
    # كل ما ليس حرفاً أو رقماً يصبح _ ثم تُزال _ من الطرفين
    out.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in out.columns]
    # تصحيح خطأ إملائي في الاسم الأصلي: nacionality ← nationality
    out.columns = [c.replace("nacionality", "nationality") for c in out.columns]
    # حذف الأعمدة المجهولة (unnamed) الناتجة عن فواصل زائدة
    return out.loc[:, [not c.startswith("unnamed") for c in out.columns]]


# قراءة CSV باكتشاف الفاصل تلقائياً: نسخة UCI تستخدم ; ونسخة Kaggle قد تستخدم ,
def read_csv_auto(path):
    """Read a CSV file or locate the right CSV inside a Kaggle input directory."""
    path = Path(path)

    # Kaggle dataset inputs are often directories, not CSV file paths.
    if path.is_dir():
        candidates = sorted(path.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No .csv files found in dataset directory: {path}. "
                "Check the Kaggle dataset attachment and its file names."
            )

        # If there are several CSVs, prefer the one whose header matches
        # the expected student-dropout dataset columns most closely.
        def header_score(candidate):
            with open(candidate, "r", encoding="utf-8-sig") as f:
                head = f.readline()
            sep = ";" if head.count(";") > head.count(",") else ","
            cols = normalize_columns(pd.DataFrame(columns=head.rstrip("\r\n").split(sep))).columns
            return (int(TARGET in cols), len(set(EXPECTED_COLUMNS).intersection(cols)))

        path = max(candidates, key=header_score)
        print(f"Using dataset CSV: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"Dataset CSV path does not exist: {path}")

    # The original UCI file uses ';'; some Kaggle copies use ','.
    with open(path, "r", encoding="utf-8-sig") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


# تحميل البيانات الخام: قراءة، توحيد الأسماء، فحص الأعمدة، إنشاء المفتاح الاصطناعي
def load_raw(path=RAW_CSV):
    # قراءة الملف وتوحيد أسماء الأعمدة
    df = normalize_columns(read_csv_auto(path))
    # الأعمدة المتوقعة غير الموجودة في الملف
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    # إن غاب عمود نوقف التنفيذ برسالة واضحة (أول خط دفاع قبل عقد Pandera)
    if missing:
        raise ValueError(f"Columns not found after normalisation: {missing}\nFound: {list(df.columns)}")
    # تحويل الهدف إلى نص وإزالة المسافات الزائدة
    df[TARGET] = df[TARGET].astype(str).str.strip()
    # البيانات بلا معرّف طالب، فننشئ مفتاحاً اصطناعياً (surrogate key) = رقم الصف
    if "student_id" not in df.columns:   # surrogate key: the dataset has no student identifier
        df.insert(0, "student_id", np.arange(len(df), dtype="int64"))
    return df


# تحويل الهدف إلى 0/1: 1 = Dropout (الموجبة)، 0 = غير ذلك
def to_binary(target_series):
    return (target_series == POSITIVE).astype(int)


# ---- gates -------------------------------------------------------------------------
# استثناء خاص يُرفع عند فشل بوابة ليتوقف الخط
class GateError(RuntimeError):
    pass


# البوابة (Gate): تسجّل النتيجة في reports/gates.json وتطبع PASS/FAIL، وعند الفشل (strict) توقف الخط. هذا هو «Gate: the run stops» في الشريحة 26
def gate(name, passed, detail="", strict=True):
    """Record a pipeline gate; stop the run when it fails (strict)."""
    # ملف سجل البوابات
    path = REPORTS / "gates.json"
    # نقرأ السجل القديم إن وُجد ثم نحدّثه (بوابة واحدة لكل اسم)
    log = json.loads(path.read_text()) if path.exists() else {}
    # تسجيل نتيجة هذه البوابة وتفاصيلها
    log[name] = {"passed": bool(passed), "detail": str(detail)}
    path.write_text(json.dumps(log, indent=2))
    # طباعة PASS أو FAIL مع اسم البوابة وتفاصيلها
    print(("PASS" if passed else "FAIL") + f" [gate] {name}: {detail}")
    # الفشل في الوضع الصارم يوقف الخط فوراً
    if strict and not passed:
        raise GateError(f"{name}: {detail}")
    return bool(passed)


# بصمة md5 للملف تُقرأ على دفعات (1MB). تُستخدم لتتبع النسب (Lineage) وربط النموذج ببياناته
def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---- ONE feature definition for training and serving (Rule #32) ---------------------
# تعريف واحد للميزة المشتقة يستورده التدريب والخدمة معاً (Rule #32) فلا ينشأ skew. نسبة النجاح = المنجَز ÷ المسجَّل
def approval_rate(approved, enrolled):
    # تحويل المدخلات إلى مصفوفات عشرية
    approved, enrolled = np.asarray(approved, float), np.asarray(enrolled, float)
    # قسمة آمنة: إن كان المقام صفراً تكون النتيجة 0 بدل خطأ أو قيمة لانهائية
    return np.divide(approved, enrolled, out=np.zeros_like(approved), where=enrolled > 0)


# ---- ONE preprocessing + model object (P5 + P6) ------------------------------------
# بناء كائن Pipeline واحد يجمع المعالجة والنموذج؛ يُدرَّب ويُحفظ ويُحمَّل في الخدمة كما هو (P5 + P6)
def make_pipeline(features, model=None):
    # الاستيراد داخل الدالة حتى لا تُحمَّل المكتبات إلا عند الحاجة
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # الأعمدة الفئوية الموجودة ضمن الميزات المطلوبة
    cat = [c for c in ENROLL_CAT if c in features]
    # الباقي رقمي (ويشمل الثنائي)
    num = [c for c in features if c not in cat]
    # معالجة مختلفة لكل نوع من الأعمدة
    pre = ColumnTransformer([
        # للرقمي: ملء الفقد بالوسيط + أعمدة مؤشر الفقد (add_indicator) ثم توحيد القياس
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num),
        # للفئوي: One-Hot. القيمة غير المعروفة تصبح أصفاراً بدل أن تُسقط الخدمة، والفئات الأقل من 20 مشاهدة تُدمج
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
    ])
    # النموذج الافتراضي: انحدار لوجستي بأوزان متوازنة لأن الفئتين غير متساويتين
    clf = model if model is not None else LogisticRegression(max_iter=2000, class_weight="balanced")
    # كل الإحصاءات (وسيط، متوسط، فئات) تُتعلَّم من بيانات التدريب فقط عند fit
    return Pipeline([("pre", pre), ("clf", clf)])

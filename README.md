# Student Dropout Early Warning

## Six-field statement
| Field | Value |
|---|---|
| Input x | Per-student record available at t₀: enrollment and application data (application mode and order, course, attendance, previous qualification and grade, admission grade), demographics (age at enrollment, gender, marital status, nationality, displaced, special needs), family background (parents' qualification and occupation), socioeconomic and macro variables (scholarship, unemployment, inflation, GDP), and the 6 first-semester curricular columns (credited, enrolled, evaluations, approved, grade, without evaluations). Excluded: all 6 second-semester columns. |
| Output y | Binary: y = 1 if the final outcome is Dropout, y = 0 otherwise. The model outputs a calibrated probability P(dropout).|
| Prediction time t0 | End of the first semester, after first-semester results are recorded. One prediction per student, made once per cohort. |
| Decision | Academic advisors contact students above a risk threshold. They offer tutoring, a counseling meeting, or a referral to financial support, before the second semester. |
| Metric | Recall on dropouts |
| Constraints | Advisor capacity limits how many students can be flagged. Predictions must be explainable per student. Flagging must not rest on financial status alone. Student data is personal and sensitive. Batch inference is enough, with no real-time latency requirement. With only 4,424 rows, use cross-validation and report confidence intervals. The pipeline must run with one command, with all settings in params.yaml. |

## How to reproduce
(to be completed in W04: dvc repro)

"""Shared preprocessing for the Student Performance Prediction project.

Created in Step 4 (Prepare data for modeling). Import build_preprocessor()
in the modeling notebook and combine it with each model in a Pipeline so
that scaling is fitted on the training data only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

NUMERICAL_FEATURES = ["Age", "StudyTimeWeekly", "Absences"]
ORDINAL_FEATURES = ["ParentalEducation", "ParentalSupport"]
BINARY_FEATURES = [
    "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering",
]


def build_preprocessor(features=None):
    """Return an unfitted ColumnTransformer for the selected inputs.

    features: optional list of input names to keep (for example, the
    inputs without Absences). Defaults to all ten selected inputs.
    """
    def keep(columns):
        return [c for c in columns if features is None or c in features]

    return ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(),
             keep(NUMERICAL_FEATURES) + keep(ORDINAL_FEATURES)),
            ("binary", "passthrough", keep(BINARY_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

import yaml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class BaselineModels:
    def __init__(self, config_path: str = "configs/baseline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["models"]
            
    def get_logistic_regression(self) -> Pipeline:
        conf = self.config.get("logistic_regression", {})
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=conf.get("max_iter", 2000),
                class_weight=conf.get("class_weight", "balanced"),
                random_state=self.config.get("random_state", 42)
            ))
        ])

    def get_svm(self) -> Pipeline:
        conf = self.config.get("svm", {})
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel=conf.get("kernel", "rbf"),
                C=conf.get("C", 1.0),
                gamma=conf.get("gamma", "scale"),
                class_weight=conf.get("class_weight", "balanced"),
                probability=conf.get("probability", True),
                random_state=self.config.get("random_state", 42)
            ))
        ])

    def get_random_forest(self) -> Pipeline:
        conf = self.config.get("random_forest", {})
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=conf.get("n_estimators", 300),
                min_samples_leaf=conf.get("min_samples_leaf", 2),
                class_weight=conf.get("class_weight", "balanced"),
                n_jobs=conf.get("n_jobs", -1),
                random_state=self.config.get("random_state", 42)
            ))
        ])
        
    def get_all_models(self) -> dict:
        return {
            "Logistic Regression": self.get_logistic_regression(),
            "SVM": self.get_svm(),
            "Random Forest": self.get_random_forest()
        }

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class ModelFactory:
    """
    Factory class to create and train various models.
    """
    
    @staticmethod
    def get_model(model_name, params=None):
        if params is None:
            params = {}
            
        if model_name == 'LogisticRegression':
            return LogisticRegression(fit_intercept=True, max_iter=10000, random_state=42, **params)
        elif model_name == 'DecisionTree':
            return DecisionTreeClassifier(random_state=42, **params)
        elif model_name == 'RandomForest':
            return RandomForestClassifier(random_state=42, **params)
        elif model_name == 'GradientBoosting':
            return GradientBoostingClassifier(random_state=42, **params)
        elif model_name == 'XGBoost':
            return XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss', **params)
        elif model_name == 'KNN':
            return KNeighborsClassifier(**params)
        elif model_name == 'NaiveBayes':
            return GaussianNB(**params)
        elif model_name == 'SVM':
            return SVC(probability=True, random_state=42, **params)
        else:
            raise ValueError(f"Model {model_name} not supported.")

    @staticmethod
    def train_with_cv(model, X_train, y_train, param_grid):
        """
        Trains a model using GridSearchCV.
        """
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        return grid_search

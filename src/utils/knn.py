from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

class KNNEvaluator():
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.knn_0 = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        self.knn_1 = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')

    def fit(self, X_control=None, y_control=None, X_treated=None, y_treated=None):
        if X_control is not None and y_control is not None:
            self.knn_0.fit(X_control, y_control)
        if X_treated is not None and y_treated is not None:
            self.knn_1.fit(X_treated, y_treated)

    def predict(self, X, predict_cf_control):
        if predict_cf_control:
            return self.knn_1.predict(X)
        else:
            return self.knn_0.predict(X)
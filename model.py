"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
def load_housing():
    url="https://github.com/ageron/data/raw/main/housing.tgz"
    tar_path=os.path.join(tempfile.gettempdir(),"housing.tgz")
    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(url,tar_path)
    with tarfile.open(tar_path,mode="r:gz") as housing_tar:
        csv_file=housing_tar.extractfile("housing/housing.csv")
        if csv_file is None:
            raise FileNotFoundError("housing/housing.csv not found in the archive")
        df = pd.read_csv(csv_file)
    return df

    pass

# Step 2 - income_categories
def income_categories(df):
    inc_cat=pd.cut(df["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    return inc_cat.astype(int)
    pass

# Step 3 - stratified_split
import sklearn
def stratified_split(df, test_size=0.2, random_state=42):
    train_set,test_set=sklearn.model_selection.train_test_split(df,test_size=test_size,random_state=random_state,stratify=income_categories(df))
    return train_set,test_set;
    pass

# Step 4 - explore_correlations
def explore_correlations(df):
    corr_mat=df.corr(method='pearson',numeric_only=True)
    correl=corr_mat["median_house_value"]
    correl=correl.drop("median_house_value")
    return correl.sort_values(ascending=False)
    pass

# Step 5 - add_ratio_features
def add_ratio_features(df):
    df_copy=df.copy()
    df_copy["rooms_per_house"]=df["total_rooms"]/df["households"]
    df_copy["bedrooms_ratio"]=df["total_bedrooms"]/df["total_rooms"]
    df_copy["people_per_house"]=df["population"]/df["households"]
    return df_copy
    pass

# Step 6 - split_features_labels
def split_features_labels(df):
    df_new=df.copy()
    X=df_new.drop("median_house_value",axis=1)
    y=df["median_house_value"]
    return X,y
    pass

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters=n_clusters
        self.gamma=gamma
        self.random_state=random_state
        pass

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_=KMeans(n_clusters=self.n_clusters,n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X,sample_weight=sample_weight)
        return self
        pass

    def transform(self, X):
        return rbf_kernel(X,self.kmeans_.cluster_centers_,gamma=self.gamma)
        pass

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
        pass

# Step 8 - numeric_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pass

# Step 9 - categorical_pipeline
from sklearn.preprocessing import OneHotEncoder
def categorical_pipeline():
    return make_pipeline(SimpleImputer(strategy="most_frequent",missing_values=None),OneHotEncoder(handle_unknown="ignore"))
    pass

# Step 10 - build_preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    num_val=["total_bedrooms", "total_rooms", "population", "households", "median_income"]
    log_pipe=Pipeline(steps=[('num_imputer',SimpleImputer(strategy="median")),('log_trans',FunctionTransformer(np.log, feature_names_out="one-to-one")),("num_stand",StandardScaler())])
    preprocessor=ColumnTransformer(transformers=[('log',log_pipe,num_val),('geo',ClusterSimilarity(n_clusters,gamma,random_state),["latitude", "longitude"]),('cat',categorical_pipeline(),["ocean_proximity"])],remainder=numeric_pipeline())
    return preprocessor
    pass

# Step 11 - rmse
from sklearn.metrics import root_mean_squared_error as rms
def rmse(y_true, y_pred):
    return rms(y_pred,y_true)
    pass

# Step 12 - dummy_baseline_rmse
from sklearn.dummy import DummyRegressor
def dummy_baseline_rmse(X, y):
    dum_reg=DummyRegressor(strategy="mean")
    dum_reg.fit(X,y)
    y_pred=dum_reg.predict(X)
    return rmse(y_pred,y)
    pass

# Step 13 - cross_val_rmse
from sklearn.model_selection import cross_val_score

def cross_val_rmse(model, X, y, cv=3):
    scores = -cross_val_score(
        model, X, y, cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    scores=scores.astype(float)
    return {"scores":scores.tolist(),"mean":float(np.mean(scores)),"std":float(np.std(scores))}

# Step 14 - linear_model
from sklearn.linear_model import LinearRegression
def linear_model(preprocessing):
    return make_pipeline(preprocessing,LinearRegression())
    pass

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement


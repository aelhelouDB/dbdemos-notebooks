# Databricks notebook source
# MAGIC %md
# MAGIC # Execute a Hyper-Paramater tuning run
# MAGIC
# MAGIC We'll run a couple of trainings to test different models
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-2.png?raw=True" width="1200">
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffeatures%2Fmlops%2F02_auto_ml&dt=MLOPS">
# MAGIC <!-- [metadata={"description":"MLOps end2end workflow: Auto-ML notebook",
# MAGIC  "authors":["quentin.ambard@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["auto-ml"]},
# MAGIC                  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering lightgbm mlflow optuna optuna-integration shap
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false $adv_mlops=true

# COMMAND ----------

import mlflow


# Path defined in the init notebook
mlflow.set_experiment(f"{xp_path}/{xp_name}")
print(f"Set experiment to: {xp_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC Load data directly from feature store
# MAGIC

# COMMAND ----------

display(spark.table("advanced_churn_feature_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We'll also use specific feature functions for on-demand features.
# MAGIC
# MAGIC Recall that we have defined the `avg_price_increase` feature function in the [feature engineering notebook]($./01_feature_engineering)

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE FUNCTION avg_price_increase

# COMMAND ----------

# MAGIC %md
# MAGIC Create feature specifications.
# MAGIC
# MAGIC The feature lookup definition specifies the tables to use as feature tables and the key to lookup feature values.
# MAGIC
# MAGIC The feature function definition specifies which columns from the feature table are bound to the function inputs.
# MAGIC
# MAGIC The Feature Engineering client will use these values to create a training specification that's used to assemble the training dataset from the labels table and the feature table.

# COMMAND ----------

# DBTITLE 1,Define feature lookups
from databricks.feature_store import FeatureFunction, FeatureLookup


feature_lookups_n_functions = [
    FeatureLookup(
      table_name=f"{catalog}.{db}.advanced_churn_feature_table",
      lookup_key=["customer_id"],
      timestamp_lookup_key="transaction_ts"
    ),
    FeatureFunction(
      udf_name=f"{catalog}.{db}.avg_price_increase",
      input_bindings={
        "monthly_charges_in" : "monthly_charges",
        "tenure_in" : "tenure",
        "total_charges_in" : "total_charges"
      },
      output_name="avg_price_increase"
    )
]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Read the label table.

# COMMAND ----------

# DBTITLE 1,Pull labels to use for training/validating/testing
labels_df = spark.read.table(f"advanced_churn_label_table")

# Set variable for label column. This will be used within the training code.
label_col = "churn"
pos_label = "Yes"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Create the training set specifications. This contains information on how the training set should be assembled from the label table, feature table, and feature function.

# COMMAND ----------

# DBTITLE 1,Create Training specifications
from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()

# Create Feature specifications object
training_set_specs = fe.create_training_set(
  df=labels_df, # DataFrame with lookup keys and label/target (+ any other input)
  label="churn",
  feature_lookups=feature_lookups_n_functions,
  exclude_columns=["customer_id", "transaction_ts", 'split']
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC With the training set specification, we can now build the training dataset.
# MAGIC
# MAGIC `training_set_specs.load_df()` returns a pySpark dataframe. We will convert it to a Pandas dataframe to train an LGBM model.

# COMMAND ----------

# DBTITLE 1,Load training set as Pandas dataframe
training_pdf = training_set_specs.load_df().toPandas()
X, Y = (training_pdf.drop(label_col, axis=1), training_pdf[label_col])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boolean columns
# MAGIC For each column, impute missing values and then convert into ones and zeros.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


bool_imputers = []

bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
    ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
])

bool_transformers = [("boolean", bool_pipeline, ["gender", "phone_service", "dependents", "senior_citizen", "paperless_billing", "partner"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with the mean by default.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler


num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["avg_price_increase", "monthly_charges", "num_optional_services", "tenure", "total_charges"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["monthly_charges", "total_charges", "avg_price_increase", "tenure", "num_optional_services"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

one_hot_imputers = []
one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", SklearnOneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["contract", "device_protection", "internet_service", "multiple_lines", "online_backup", "online_security", "payment_method", "streaming_movies", "streaming_tv", "tech_support"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer


transformers = bool_transformers + numerical_transformers + categorical_one_hot_transformers
preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scaling Hyper-Parameter-Optimization using Optuna on a single node
# MAGIC
# MAGIC Optuna is an advanced hyperparameter optimization framework designed specifically for machine learning tasks. Here are the key ways Optuna conducts hyperparameter optimization primarly via:
# MAGIC
# MAGIC 1. Sampling Algorithms:
# MAGIC     1. **Tree-structured Parzen Estimator (Default)** - Bayesian optimization to efficiently search the hyperparameter space.
# MAGIC     2. Random Sampling - Randomly samples hyperparameter values from the search space.
# MAGIC     3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - An evolutionary algorithm for difficult non-linear non-convex optimization problems.
# MAGIC     4. Grid Search - Exhaustively searches through a manually specified subset of the hyperparameter space.
# MAGIC     5. Quasi-Monte Carlo sampling - Uses low-discrepancy sequences to sample the search space more uniformly than pure random sampling.
# MAGIC     6. NSGA-II (Non-dominated Sorting Genetic Algorithm II) - A multi-objective optimization algorithm.
# MAGIC     7. Gaussian Process-based sampling (i.e. Kriging) - Uses Gaussian processes for Bayesian optimization.
# MAGIC     8. Optuna also allows implementing custom samplers by inheriting from the `BaseSampler` class.
# MAGIC 2. Pruning: Optuna implements pruning algorithms to early-stop unpromising trials
# MAGIC 3. Study Object: Users create a "study" object that manages the optimization process. The study.optimize() method is called to start the optimization, specifying the objective function and number of trials.
# MAGIC 4. Parallel Execution: Optuna scales to parallel execution on a single node _(for multinode scaling please refer to this [repo](https://github.com/databricks-industry-solutions/ray-framework-on-databricks/tree/main/Hyperparam_Optimization))_.
# MAGIC
# MAGIC
# MAGIC We'll leverage Optuna's native MLflow integration: the `MLflowCallback` which helps automatically logging the hyperparameters and metrics.
# MAGIC
# MAGIC Then we'll run an Optuna hyperparameter optimization study by passing the `MLflowCallback` object to the optimize function.

# COMMAND ----------

import optuna


optuna_sampler = optuna.samplers.TPESampler(
  n_startup_trials=4, # If using on-demand cluster you can set to num_cpu_cores in driver node (or more if spark is oversubscribed)
  n_ei_candidates=4, # If using on-demand cluster you can set to num_cpu_cores in driver node (or more if spark is oversubscribed)
  seed=2025 # Random Number Generator seed
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define objective/loss function and search space to optimize
# MAGIC The search space here is defined by calling functions such as `suggest_categorical`, `suggest_float`, `suggest_int` for the Trial object that is passed to the objective function. Optuna allows to define the search space dynamically.
# MAGIC
# MAGIC Refer to the documentation for:
# MAGIC * [optuna.samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) for the choice of samplers
# MAGIC * [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) for a full list of functions supported to define a hyperparameter search space.

# COMMAND ----------

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class ObjectiveOptuna(object):
  """
  a callable class for implementing the objective function. It takes the training dataset by a constructor's argument
  instead of loading it in each trial execution. This will speed up the execution of each trial
  """
  def __init__(self, X_train_in:pd.DataFrame, Y_train_in:pd.Series, preprocessor_in: ColumnTransformer, rng_seed:int=2025, pos_label_in:str=pos_label):
    """
    X_train_in: features
    Y_train_in: label
    """

    # Set pre-processing pipeline
    self.preprocessor = preprocessor_in
    self.rng_seed = rng_seed
    self.pos_label = pos_label_in

    # Split into training and validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_in, Y_train_in, test_size=0.1, random_state=rng_seed)

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_val = X_val
    self.Y_val = Y_val
    
  def __call__(self, trial):
    """
    Wrapper call containing data processing pipeline, training and hyperparameter tuning code.
    The function returns the weighted F1 accuracy metric to maximize in this case.
    """

    # Define list of classifiers to test
    classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "RandomForest", "LightGBM"])
    
    if classifier_name == "LogisticRegression":
      # Optimize tolerance and C hyperparameters
      lr_C = trial.suggest_float("C", 1e-2, 1, log=True)
      lr_tol = trial.suggest_float('tol' , 1e-6 , 1e-3, step=1e-6)
      classifier_obj = LogisticRegression(C=lr_C, tol=lr_tol, random_state=self.rng_seed)

    elif classifier_name == "RandomForest":
      # Optimize number of trees, tree depth, min_sample split and leaf hyperparameters
      n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
      max_depth = trial.suggest_int("max_depth", 3, 10)
      min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
      min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
      classifier_obj = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=self.rng_seed)

    elif classifier_name == "LightGBM":
      # Optimize number of trees, tree depth, learning rate and maximum number of bins hyperparameters
      n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
      max_depth = trial.suggest_int("max_depth", 3, 10)
      learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.9)
      max_bin = trial.suggest_int("max_bin", 2, 256)
      num_leaves = trial.suggest_int("num_leaves", 2, 256),
      classifier_obj = LGBMClassifier(force_row_wise=True, verbose=-1,
                                      n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, max_bin=max_bin, num_leaves=num_leaves, random_state=self.rng_seed)
    
    # Assemble the pipeline
    this_model = Pipeline(steps=[("preprocessor", self.preprocessor), ("classifier", classifier_obj)])

    # Fit the model
    mlflow.sklearn.autolog(disable=True) # Disable mlflow autologging to avoid logging artifacts for every run
    this_model.fit(self.X_train, self.Y_train)

    # Predict on validation set
    y_val_pred = this_model.predict(self.X_val)

    # Calculate and return F1-Score
    f1_score_binary= f1_score(self.Y_val, y_val_pred, average="binary", pos_label=self.pos_label)

    return f1_score_binary

# COMMAND ----------

# MAGIC %md
# MAGIC Quick test/debug locally

# COMMAND ----------

objective_fn = ObjectiveOptuna(X, Y, preprocessor, pos_label_in=pos_label)
study_debug = optuna.create_study(direction="maximize", study_name="test_debug", sampler=optuna_sampler)
study_debug.optimize(objective_fn, n_trials=2)

# COMMAND ----------

print("Best trial:")
best_trial = study_debug.best_trial
print(f"  F1_score: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add `MLflowCallback`, wrap training function and execute as part of parent/child mlflow run

# COMMAND ----------

from optuna.integration.mlflow import MLflowCallback


# Grab experiment and model name
experiment_name = f"{xp_path}/{xp_name}"

try:
  experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

except:
  with mlflow.start_run(run_name="dummy-run"):
    # Dummy run to create notebook experiment if it doesn't exist
    mlflow.end_run()
  experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Creae Optuna-Native MLflow Callback
mlflc = MLflowCallback(
    tracking_uri="databricks",
    metric_name="f1_score_val",
    create_experiment=False,
    mlflow_kwargs={
        "experiment_id": experiment_id,
        "nested":True
    }
)

# COMMAND ----------

import warnings
import pandas as pd
from mlflow.types.utils import _infer_schema
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc


def optuna_hpo_fn(n_trials: int, features: pd.DataFrame, labels: pd.Series, training_set_specs_in, preprocessor_in: ColumnTransformer, experiment_id: str, include_mlflc: bool, pos_label_in: str = pos_label, rng_seed_in: int = 2025) -> optuna.study.study.Study:
    # Start mlflow run
    with mlflow.start_run(run_name="mlops_churn_hpo", experiment_id=experiment_id) as parent_run:

        # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=rng_seed_in)

        # Kick distributed HPO as nested runs
        objective_fn = ObjectiveOptuna(X_train, Y_train, preprocessor_in, rng_seed_in, pos_label_in)
        optuna_study = optuna.create_study(direction="maximize", study_name=f"mlops_churn_hpo_study", sampler=optuna_sampler)
        if include_mlflc:
            optuna_study.optimize(objective_fn, n_trials=n_trials, callbacks=[mlflc])
        else:
            optuna_study.optimize(objective_fn, n_trials=n_trials)

        # Extract best trial info
        best_model_params = optuna_study.best_params
        best_model_params["random_state"] = rng_seed_in
        classifier_type = best_model_params.pop('classifier')

        # Reproduce best classifier
        if classifier_type  == "LogisticRegression":
            best_model = LogisticRegression(**best_model_params)
        elif classifier_type == "RandomForestClassifier":
            best_model = RandomForestClassifier(**best_model_params)
        elif classifier_type == "LightGBM":
            best_model = LGBMClassifier(force_row_wise=True, verbose=-1, **best_model_params)

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, log_models=False, silent=True)
        
        # Fit best model and log using FE client in parent run
        model_pipeline = Pipeline(steps=[("preprocessor", objective_fn.preprocessor), ("classifier", best_model)])
        model_pipeline.fit(X_train, Y_train)
        
        fe.log_model(
            model=model_pipeline,
            artifact_path="model",
            flavor=mlflow.sklearn,
            training_set=training_set_specs_in,
        )

        # Evaluate model and log into experiment
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model_pipeline)
        
        # Log metrics for the training set
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(label_col):Y_train}),
            targets=label_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "training_" , "pos_label": pos_label_in }
        )

        # Log metrics for the test set
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(label_col):Y_test}),
            targets=label_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "test_" , "pos_label": pos_label_in }
        )

        mlflow.end_run()
        
        return optuna_study

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute training runs

# COMMAND ----------

n_trials: int, features: pd.DataFrame, labels: pd.Series, training_set_specs_in, preprocessor_in: ColumnTransformer, experiment_id: str, include_mlflc: bool, pos_label_in: str = pos_label, rng_seed_in: int = 2025

# COMMAND ----------

# Disable mlflow autologging to minimize overhead
# mlflow.autolog(disable=True) # Disable mlflow autologging

# Setting the logging level DEBUG to avoid too verbose logs
optuna.logging.set_verbosity(optuna.logging.DEBUG)
optuna.logging.disable_propagation()

# Invoke training function on driver node
single_node_study = optuna_hpo_fn(
  n_trials=16,
  features=X,
  labels=Y,
  training_set_specs_in=training_set_specs,
  preprocessor_in=preprocessor,
  experiment_id=experiment_id,
  include_mlflc=True,
  pos_label_in=pos_label,
  rng_seed_in=2025
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, so we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

# COMMAND ----------

if shap_enabled:
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    from shap import KernelExplainer, summary_plot
    # SHAP cannot explain models using data with nulls.
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
    mode = X_train.mode().iloc[0]

    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=790671489).fillna(mode)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=790671489).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="logit")
    shap_values = explainer.shap_values(example, l1_reg=False, nsamples=100)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC, and Precision-Recall curves for validation data
# MAGIC
# MAGIC We show the confusion matrix, RO,C and Precision-Recall curves of the model on the validation data.
# MAGIC
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Click the link to see the MLflow run page
displayHTML(f"<a href=#mlflow/experiments/{mlflow_run.info.experiment_id}/runs/{ mlflow_run.info.run_id }/artifactPath/model> Link to model run page </a>")

# COMMAND ----------

import uuid
from IPython.display import Image

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve_plot.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve_plot.png")
display(Image(filename=eval_pr_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automate model promotion validation
# MAGIC
# MAGIC Next step: [Search runs and trigger model promotion validation]($./03_from_notebook_to_models_in_uc)

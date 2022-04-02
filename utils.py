import streamlit as st
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor

@st.cache
def train_model():
    predictor = TabularPredictor(label=label, eval_metric=metric)
    predictor.fit(train_data=df_train, presets='best_quality', time_limit=30,verbosity=0)
    return predictor

@st.cache
def get_summary(predictor):
    return predictor.fit_summary()

def automl_feature_enginering(df):
    tab_dataset = TabularDataset(df)
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
    auto_ml_pipeline_feature_generator.fit(X=tab_dataset)
    return auto_ml_pipeline_feature_generator
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.datasets import load_iris

@st.cache
def train_model():
    predictor = TabularPredictor(label=label, eval_metric=metric)
    predictor.fit(train_data=df_train, presets='best_quality', time_limit=30,verbosity=0)
    return predictor

@st.cache
def get_summary():
    return train_model().fit_summary()

def automl_feature_enginering(df):
    tab_dataset = TabularDataset(df)
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
    auto_ml_pipeline_feature_generator.fit(X=tab_dataset)
    return auto_ml_pipeline_feature_generator

uploaded_file = st.file_uploader("Upload your Dataset", type=['csv'])
if uploaded_file:
    iris_data =  load_iris(as_frame=True)
    df = pd.read_csv(uploaded_file)

    st.write("here is your Dataset first five rows")
    st.write(df.head())
    #Limpando o dataset antes do treino
    st.write("Data Clean")
    # columns_to_drop = st.multiselect("Selecione as colunas que deseja remover", df.columns)
    #Dividir em treino e teste
    test_size = st.number_input("Selecione a porcentagem do dataset de validação [0-100]", min_value=0, max_value=100, value=25)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    #Seleção de Label
    label = st.selectbox( 'Which column is the label?', [None] + list(df.columns),index=0)

    #Seleção de problema e metricas
    problem_type = st.radio("Classification or Regression Problem?", [None,"Classification","Regression"])
    if problem_type == 'Classification':
        metrics = [None,'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
        'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro', 'precision_micro',
        'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score']
    else:
        metrics = [None,'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2']
    
    metric = st.selectbox('Which metric do you want to use',metrics)
    #Definição do Preditor
    treinar = st.button('Iniciar Treino')
    if label and problem_type and metric and treinar:
        predictor = train_model()
        predictions = predictor.evaluate(df_test)
        st.write(predictions)
            
    if label and problem_type :
        summary = get_summary()
        st.json(summary)
        st.write("Melhores Modelos")
        st.dataframe(summary["leaderboard"])

        model = st.selectbox('Hyperparametros do Modelo', summary["model_hyperparams"].keys())
        st.json(summary["model_hyperparams"][model])
        
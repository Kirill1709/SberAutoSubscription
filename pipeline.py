import dill
import datetime

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from loguru import logger


columns_to_drop = ['client_id', 'utm_source', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_category',
                    'device_os', 'device_brand', 'device_model', 'device_browser', 'session_id',
                    'visit_date', 'visit_time', 'device_screen_resolution', 'geo_country', 'geo_city']
target_list = ['sub_car_claim_click', 'sub_car_claim_submit_click',
               'sub_open_dialog_click', 'sub_custom_question_submit_click',
               'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
               'sub_car_request_submit_click']


def drop_columns(df, columns_to_drop):
    df_copy = df.copy()
    return df_copy.drop(columns_to_drop, axis=1)


def add_features(df):
    def check_city(data):
        if data.geo_city == 'Moscow':
            return 1
        elif data.geo_city == 'Saint Petersburg':
            return 2
        elif data.country_type == 1:
            return 3
        return 4
    df['visit_month'] = df.visit_date.apply(lambda x: int(x.split('-')[1]))
    df['visit_day'] = df.visit_date.apply(lambda x: int(x.split('-')[2]))
    df['visit_weekday'] = df.visit_date.apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").weekday())
    df['visit_hour'] = df.visit_time.apply(lambda x: int(x.split(':')[0]))
    df['device_screen_width'] = df.device_screen_resolution.apply(lambda x: int(x.split('x')[0]))
    df['device_screen_height'] = df.device_screen_resolution.apply(lambda x: int(x.split('x')[1]))
    df['country_type'] = df.geo_country.apply(lambda x: 1 if x == 'Russia' else 0)
    df['city_type'] = df.apply(lambda x: check_city(x), axis=1)
    return df


def merge_df(df_hits, df_sessions):
    def check_target(actions):
        return int(any(action in target_list for action in actions))

    df_hits['target'] = df_hits['event_action'].isin(target_list)
    grouped = df_hits.groupby('session_id')['target'].any().astype(int)
    merged_df = pd.merge(df_sessions, grouped, on='session_id', how='inner')
    return merged_df


def replace_value(df, value_1, value_2):
    return df.replace(value_1, value_2)


def main():
    logger.info('Начинаю обучение')
    df_hits = pd.read_csv('data/ga_hits.csv')
    df_sessions = pd.read_csv('data/ga_sessions.csv')
    merged_df = merge_df(df_hits, df_sessions)
    X = merged_df.drop('target', axis=1)
    y = merged_df['target']
    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, ['utm_medium'])
    ])
    best_score = 0
    best_pipe = None
    models = [
        LogisticRegression(random_state=42, class_weight='balanced'),
        RandomForestClassifier(random_state=42, n_estimators=400, min_samples_leaf=9, max_features='sqrt', max_depth=20),
        MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(50,))
    ]
    for model in models:
        logger.info(type(model).__name__)
        pipe = Pipeline(steps=[
            ('replace_value', FunctionTransformer(replace_value, kw_args={'value_1': '(none)', 'value_2': '(not set)'})),
            ('add_features', FunctionTransformer(add_features)),
            ('drop_columns', FunctionTransformer(drop_columns, kw_args={'columns_to_drop': columns_to_drop})),
            ('preprocessor', preprocessor),
            ('classifier', model),
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        logger.info(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        if score.mean() > 0.65:
            if score.mean() > best_score:
                best_score = score.mean()
                best_pipe = pipe
    if best_score == 0:
        logger.info('Ни одна из моделей не достигла требуемой точности')
        return
    logger.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    best_pipe.fit(X, y)
    with open('car_subscription.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Price category prediction model',
                'author': 'Kirill Gatsuk',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file, recurse=True)



if __name__ == '__main__':
    main()
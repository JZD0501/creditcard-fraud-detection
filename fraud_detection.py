import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ========== 全局路径设置 ==========
PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / 'creditcard.csv'
MODEL_DIR = PROJECT_DIR / 'models'
FIG_DIR = PROJECT_DIR / 'figures'
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ========== 字体配置 ==========
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 加载数据 ==========
def load_creditcard_data(data_path=DATA_PATH):
    try:
        df = pd.read_csv(data_path)
        print(f"数据集加载完成，形状: {df.shape}")
        print(f"欺诈交易占比: {df['Class'].mean():.4%}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

# ========== 数据探索 ==========
def explore_data(df):
    print("\n数据集无缺失值" if df.isnull().sum().sum() == 0 else df.isnull().sum())
    #print("\n数据类型:")
    #print(df.dtypes)

    # 对金额取对数，避免偏态严重
    df['LogAmount'] = np.log1p(df['Amount'])

    # 分图展示正常和欺诈交易的金额分布
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue')
    plt.title('正常交易金额分布')
    plt.xlabel('交易金额')
    plt.ylabel('频数')

    plt.subplot(1, 2, 2)
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=30, color='red')
    plt.title('欺诈交易金额分布')
    plt.xlabel('交易金额')
    plt.ylabel('频数')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'amount_distribution_split.png')
    plt.close()

    # 特征相关性（仅 V1-V28 + Class）
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Class']
    plt.figure(figsize=(10, 6))
    corr = df[feature_cols].corr()['Class'].sort_values(ascending=False)[1:11]
    sns.barplot(x=corr.values, y=corr.index)
    plt.title('与欺诈相关性最高的10个特征')
    plt.savefig(FIG_DIR / 'feature_correlation.png')
    plt.close()

    # 热力图：仅 V1-V28 + Class
    plt.figure(figsize=(14, 12))
    sns.heatmap(df[feature_cols].corr(), cmap='coolwarm', center=0, annot=False, fmt='.2f')
    plt.title('特征间相关系数热力图')
    plt.savefig(FIG_DIR / 'correlation_heatmap.png')
    plt.close()

    return df

# ========== 线性回归 ==========
def linear_regression_task(df):
    normal_df = df[df['Class'] == 0]
    X = normal_df.drop(['Time', 'Amount', 'Class'], axis=1)
    y = normal_df['Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("\n线性回归（交易金额预测）评估结果:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': np.abs(model.coef_)
    }).sort_values('重要性', ascending=False)
    print("\n对交易金额影响最大的前5个特征:")
    print(importance.head())

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('真实金额')
    plt.ylabel('预测金额')
    plt.title('线性回归预测效果')
    plt.savefig(FIG_DIR / 'linear_regression_prediction.png')
    plt.close()
    return model, scaler, importance

# ========== 逻辑回归 ==========
def train_logistic_with_sampling(X_train, y_train, method):
    if method == 'original':
        return LogisticRegression(solver='liblinear', random_state=42).fit(X_train, y_train)
    elif method == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        return LogisticRegression(solver='liblinear', random_state=42).fit(X_res, y_res)
    elif method == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return LogisticRegression(solver='liblinear', random_state=42).fit(X_res, y_res)

def logistic_regression_task(df):
    X = df.drop(['Time', 'Amount', 'Class'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        '原始': train_logistic_with_sampling(X_train_scaled, y_train, 'original'),
        '欠采样': train_logistic_with_sampling(X_train_scaled, y_train, 'undersample'),
        'SMOTE过采样': train_logistic_with_sampling(X_train_scaled, y_train, 'smote')
    }

    print("\n逻辑回归（欺诈检测）AUC评估结果:")
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"{name}: {auc:.4f}")

    # 可视化ROC
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC 曲线对比')
    plt.legend()
    plt.savefig(FIG_DIR / 'roc_comparison.png')
    plt.close()

    # 打印三种模型的分类报告
    print("\n三种采样方法下的分类报告对比:")
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        print(f"\n==== {name} ====")
        print(classification_report(y_test, y_pred, target_names=['正常交易', '欺诈交易']))

    # 使用 SMOTE 模型为后续输出特征重要性
    best_model = models['SMOTE过采样']
    importance = pd.DataFrame({
        '特征': X.columns,
        '系数': best_model.coef_[0]
    }).sort_values('系数', ascending=False)
    print("\n对欺诈检测影响最大的前5个特征:")
    print(importance.head())

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='系数', y='特征', data=importance.head(10))
    plt.title('逻辑回归：特征重要性')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'logistic_feature_importance.png')
    plt.close()

    return best_model, scaler, importance

# ========== 模型保存 ==========
def save_models(lr_model, clf_model, lr_scaler, clf_scaler):
    joblib.dump(lr_model, MODEL_DIR / 'linear_regression_model.pkl')
    joblib.dump(clf_model, MODEL_DIR / 'logistic_regression_model.pkl')
    joblib.dump(lr_scaler, MODEL_DIR / 'scaler_reg.pkl')
    joblib.dump(clf_scaler, MODEL_DIR / 'scaler_clf.pkl')
    print("\n模型保存完成 ")

# ========== 主函数 ==========
def main():
    df = load_creditcard_data()
    if df is None:
        return
    df = explore_data(df)
    lin_model, lin_scaler, lin_importance = linear_regression_task(df)
    log_model, log_scaler, log_importance = logistic_regression_task(df)
    save_models(lin_model, log_model, lin_scaler, log_scaler)

if __name__ == '__main__':
    main()

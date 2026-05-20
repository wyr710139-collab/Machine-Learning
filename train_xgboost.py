import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("錯誤：找不到 xgboost 函式庫。")
    print("請使用 'pip install xgboost' 指令安裝它。")
    sys.exit(1)


# --- 路徑設定 ---
# 將父目錄添加到 Python 的模組搜尋路徑中
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 匯入自訂模組 ---
try:
    # 從父目錄的 model.py 匯入評估函式
    from model import evaluate_regression_model
    from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
except ImportError:
    print("錯誤：無法從父目錄找到 'model.py' 或 'evaluate_regression_model' 函式。")
    print("請確認 'model.py' 檔案與函式名稱是否正確。")
    sys.exit(1)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    繪製給定估計器的學習曲線。
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_residual_analysis(model, X_train, y_train, X_test, y_test):
    """
    進行殘差分析並繪製殘差分布圖。
    """
    # 訓練集和測試集的預測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 計算殘差
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # 繪製殘差分布直方圖
    plt.figure(figsize=(10, 6))
    plt.hist(test_residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    # 顯示殘差統計資訊
    mean_residual = np.mean(test_residuals)
    std_residual = np.std(test_residuals)
    plt.text(0.02, 0.98, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return plt

def plot_feature_importance(model, X_train, top_n=5):
    """
    繪製特徵重要性分析圖，只顯示最重要的前 N 個特徵。
    """
    # 獲取特徵重要性
    importance = model.feature_importances_
    feature_names = X_train.columns
    
    # 創建 DataFrame 並排序
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # 只取前 N 個最重要的特徵
    top_features = importance_df.head(top_n).sort_values('Importance', ascending=True)
    
    # 繪製水平條形圖
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f'XGBoost - Top {top_n} Most Important Features')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # 打印完整重要性排名
    print(f"\n=== 前 {top_n} 名特徵重要性 ===")
    for idx, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        print(f"{idx}. {row['Feature']}: {row['Importance']:.4f}")
    
    return plt

def main():
    """
    主函式：載入資料、訓練 XGBoost 模型並評估。
    """
    print("===== 開始執行 XGBoost 模型訓練腳本 ====")

    # --- 1. 載入資料 ---
    try:
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")
        y_train = pd.read_csv("y_train.csv").squeeze('columns')
        y_test = pd.read_csv("y_test.csv").squeeze('columns')
        print("成功從 CSV 檔案載入訓練與測試資料。")
        print(f"特徵數量: {X_train.shape[1]}")
    except FileNotFoundError as e:
        print(f"\n錯誤：在父目錄中找不到資料檔案 ({e.filename})。")
        print("請先完整執行 'food.py' 來生成 CSV 檔案。")
        return

    # --- 2. 設定超參數網格 ---
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
    }

    # --- 3. 使用網格搜尋進行超參數調整 ---
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # 3 折交叉驗證
        n_jobs=-1,
        verbose=2,
        scoring='neg_mean_squared_error'
    )
    
    print("\n正在進行網格搜尋...")
    grid_search.fit(X_train, y_train)
    
    print("\n網格搜尋完成。")
    print(f"找到的最佳超參數: {grid_search.best_params_}")
    print(f"最佳交叉驗證分數 (MSE): {-grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # --- 4. 使用最佳模型進行評估 ---
    print("\n--- XGBoost 最佳模型評估結果 (網格搜尋) ---")
    evaluate_regression_model(
        model=best_model,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test
    )

    # --- 5. 繪製特徵重要性分析 ---
    print("\n正在分析特徵重要性...")
    plot_feature_importance(best_model, X_train)
    plt.show()

    # --- 6. 繪製殘差分析圖 ---
    print("\n正在繪製殘差分析圖...")
    plot_residual_analysis(best_model, X_train, y_train, X_test, y_test)
    plt.show()


    print("\n===== 所有實驗執行完畢 ====")

if __name__ == '__main__':
    main()

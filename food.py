import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import requests
from sklearn.model_selection import train_test_split

# 忽略警告訊息
warnings.filterwarnings('ignore')

# 設定中文字型 (避免圖表中文亂碼，視系統環境而定，若無可註解掉)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 讀取資料 =====
excel_name = r"C:\Users\Modern\Downloads\Food Delivery Time Prediction Case Study.xlsx\Food Delivery Time Prediction Case Study.xlsx"

try:
    df = pd.read_excel(excel_name)
    print("資料讀取成功！")
except FileNotFoundError:
    print("找不到檔案，請檢查 excel_name 的路徑是否正確。")
    raise

# 移除無用 ID
df = df.drop(['ID', 'Delivery_person_ID'], axis=1)

# 修正評分 (大於 5 分的修正為 5 分)
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].apply(lambda x: 5.0 if x > 5.0 else x)

# ===== 2. 預處理、抽樣與 OSRM 計算 =====
# --- 2.1 在完整資料集上移除經緯度異常值 ---
print(f"原始資料總數: {len(df)}")
original_full_count = len(df)
df_cleaned_coords = df[(df['Restaurant_latitude'] >= -90) & (df['Restaurant_latitude'] <= 90) &
                       (df['Delivery_location_latitude'] >= -90) & (df['Delivery_location_latitude'] <= 90) &
                       (df['Restaurant_longitude'] >= -180) & (df['Restaurant_longitude'] <= 180) &
                       (df['Delivery_location_longitude'] >= -180) & (df['Delivery_location_longitude'] <= 180)]
print(f"從完整資料集中，因經緯度異常移除了 {original_full_count - len(df_cleaned_coords)} 筆資料。")
print(f"清理經緯度後剩餘資料: {len(df_cleaned_coords)}")


# --- 2.2 從清理後的資料中隨機抽取 2000 筆樣本 ---
df_sample = df_cleaned_coords.sample(n=2000, random_state=42).copy()
print(f"\n已從清理後的資料中，隨機抽取 {len(df_sample)} 筆樣本進行 OSRM 計算。")


# --- 2.3 OSRM 計算經緯度距離 ---
def osrm_distance(row):
    base_url = "http://router.project-osrm.org/route/v1/driving/"
    start = f"{row['Restaurant_longitude']},{row['Restaurant_latitude']}"
    end = f"{row['Delivery_location_longitude']},{row['Delivery_location_latitude']}"
    url = f"{base_url}{start};{end}?overview=false"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                return data['routes'][0]['distance'] / 1000  # 轉為公里
        return np.nan
    except Exception:
        return np.nan

print("\n正在使用 OSRM API 計算行車距離，請稍候...")
df_sample['osrm_distance_km'] = df_sample.apply(osrm_distance, axis=1)
print("OSRM 距離計算完成。")

# 顯示前 5 筆經緯度與距離結果
print(df_sample[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'osrm_distance_km']].head())

# ===== 3. 資料清理 =====
print("\n=== 缺失值比例（%） ===")
print(df_sample.isna().mean() * 100)

# --- 3.1 移除剩餘的缺失值 (例如 OSRM 計算失敗的) ---
df_clean = df_sample.dropna()

# --- 3.2 移除其他欄位的異常值 ---
# 年齡異常值 (16~80)
df_clean = df_clean[(df_clean['Delivery_person_Age'] >= 16) & (df_clean['Delivery_person_Age'] <= 80)]

# 距離異常值 (0~100km)
df_clean = df_clean[(df_clean['osrm_distance_km'] >= 0) & (df_clean['osrm_distance_km'] <= 100)]

print(f"\n經過完整清理後，最終剩餘資料數量: {len(df_clean)}")

# ===== 4. 機器學習前處理：特徵工程與資料分割 =====
# 針對類別欄位做 One-Hot Encoding
categorical_cols = ['Type_of_order', 'Type_of_vehicle']
df_ml = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

# 移除剩下的字串欄位 (如 Weather, City 等，若要使用需另外編碼)
df_ml = df_ml.select_dtypes(exclude=['object']) 

# 準備 X 和 y
target_col = 'Time_taken(min)'
if target_col in df_ml.columns:
    X = df_ml.drop(target_col, axis=1)
    y = df_ml[target_col]
    
    # 資料分割 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n訓練集資料量: {X_train.shape[0]}，測試集資料量: {X_test.shape[0]}")

    # 存檔
    X_train.to_csv("X_train.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv("X_test.csv", index=False, encoding="utf-8-sig")
    y_train.to_csv("y_train.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv("y_test.csv", index=False, encoding="utf-8-sig")
    print("已將分割後的資料存檔完成。")
else:
    print(f"錯誤：找不到目標欄位 {target_col}")

# ===== 5. 資料視覺化 =====
cols_to_plot = ['Delivery_person_Age', 'Delivery_person_Ratings', 'osrm_distance_km', 'Time_taken(min)']

# 1. Pairplot
if set(cols_to_plot).issubset(df_clean.columns):
    sns.pairplot(df_clean[cols_to_plot])
    plt.suptitle('Pairplot of Key Numerical Features', y=1.02)
    plt.show()
# 2. Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df_clean[cols_to_plot].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 3. Boxplot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Type_of_order', y='Time_taken(min)', data=df_clean)
plt.title('Time Taken by Order Type')

plt.subplot(1, 2, 2)
sns.boxplot(x='Type_of_vehicle', y='Time_taken(min)', data=df_clean)
plt.title('Time Taken by Vehicle Type')

plt.tight_layout()
plt.show()
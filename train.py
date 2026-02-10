import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# 1. read data
print("جاري تحميل البيانات...")
df = pd.read_csv('healthcare-dataset-stroke-data.csv')  

# 2. data cleaning
df = df.drop('id', axis=1)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# إزالة القيم الغريبة في gender (فيه "Other" واحد بس عادة)
df = df[df['gender'] != 'Other']

# 3. specify features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# 4. data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"عدد الحالات الإيجابية في التدريب: {sum(y_train==1)}")
print(f"عدد الحالات السلبية في التدريب: {sum(y_train==0)}")

# 5. determine columns
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
num_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# 6. Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# 7.  Pipeline 
model = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.7)),  #  oversampling 
    ('classifier', XGBClassifier(
        n_estimators=650,           #increase tree number
        learning_rate=0.05,         # decrease
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0.2,
        scale_pos_weight=3,         
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    ))
])

# 8. training
print("\nجاري التدريب (هياخد دقيقة أو اتنين)...")
model.fit(X_train, y_train)

# 9. prediction by using threshold (important for Recall)
y_proba = model.predict_proba(X_test)[:, 1]

#  threshold 
optimal_threshold = 0.25   

y_pred_custom = (y_proba >= optimal_threshold).astype(int)
y_pred_default = model.predict(X_test)  # threshold = 0.5

# 10. results with specified threshold 
print(f"\nنتائج مع Threshold = {optimal_threshold}:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_custom):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_custom, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_custom, zero_division=0):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom, zero_division=0))

# 11. another results by modified threshold 
print(f"\nمقارنة مع Threshold الافتراضي (0.5):")
print(f"Recall: {recall_score(y_test, y_pred_default):.4f}")

# 12.  ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[np.argmin(abs(thresholds - optimal_threshold))], 
            tpr[np.argmin(abs(thresholds - optimal_threshold))], 
            color='red', s=100, label=f'Optimal Threshold = {optimal_threshold}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve مع أفضل Threshold')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 13. save model
final_model = {
    'model': model,
    'threshold': optimal_threshold
}
joblib.dump(final_model, 'stroke_model_enhanced.pkl')
print(f"\nتم حفظ النموذج المحسن مع threshold = {optimal_threshold} في stroke_model_enhanced.pkl")
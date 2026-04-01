from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# ── Train model on startup (no saved model needed) ──────────────────────────
def train_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'cleaned_data.csv')
    df = pd.read_csv(data_path)

    df_model = df.copy()
    df_model['High_Discount']     = (df_model['Discount_offered'] > 10).astype(int)
    df_model['Heavy_Package']     = (df_model['Weight_in_gms'] > 4000).astype(int)
    df_model['High_Value']        = (df_model['Cost_of_the_Product'] > 200).astype(int)
    df_model['Frequent_Customer'] = (df_model['Prior_purchases'] >= 4).astype(int)

    le = LabelEncoder()
    for col in ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']:
        df_model[col] = le.fit_transform(df_model[col])

    feature_cols = [
        'Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls',
        'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases',
        'Product_importance', 'Gender', 'Discount_offered',
        'Weight_in_gms', 'High_Discount', 'Heavy_Package',
        'High_Value', 'Frequent_Customer'
    ]

    X = df_model[feature_cols]
    y = df_model['Late_Delivery']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le, feature_cols, df

model, le, feature_cols, df = train_model()

# ── Stats for dashboard ──────────────────────────────────────────────────────
def get_stats():
    total       = len(df)
    late        = int(df['Late_Delivery'].sum())
    on_time     = total - late
    late_pct    = round(late / total * 100, 1)
    ontime_pct  = round(on_time / total * 100, 1)

    by_mode = df.groupby('Mode_of_Shipment')['Late_Delivery'].agg(['sum','count']).reset_index()
    by_mode.columns = ['mode', 'late', 'total']
    by_mode['pct'] = (by_mode['late'] / by_mode['total'] * 100).round(1)

    by_wh = df.groupby('Warehouse_block')['Late_Delivery'].agg(['sum','count']).reset_index()
    by_wh.columns = ['block', 'late', 'total']
    by_wh['pct'] = (by_wh['late'] / by_wh['total'] * 100).round(1)

    avg_discount_late   = round(df[df['Late_Delivery']==1]['Discount_offered'].mean(), 1)
    avg_discount_ontime = round(df[df['Late_Delivery']==0]['Discount_offered'].mean(), 1)
    avg_weight_late     = round(df[df['Late_Delivery']==1]['Weight_in_gms'].mean(), 0)
    avg_weight_ontime   = round(df[df['Late_Delivery']==0]['Weight_in_gms'].mean(), 0)

    return {
        'total': total, 'late': late, 'on_time': on_time,
        'late_pct': late_pct, 'ontime_pct': ontime_pct,
        'by_mode': by_mode.to_dict('records'),
        'by_warehouse': by_wh.to_dict('records'),
        'avg_discount_late': avg_discount_late,
        'avg_discount_ontime': avg_discount_ontime,
        'avg_weight_late': int(avg_weight_late),
        'avg_weight_ontime': int(avg_weight_ontime),
    }

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', stats=get_stats())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        warehouse_map    = {'A':0,'B':1,'C':2,'D':3,'F':4}
        shipment_map     = {'Flight':0,'Road':1,'Ship':2}
        importance_map   = {'high':0,'low':1,'medium':2}
        gender_map       = {'F':0,'M':1}

        warehouse   = warehouse_map.get(data['warehouse_block'], 0)
        shipment    = shipment_map.get(data['mode_of_shipment'], 2)
        importance  = importance_map.get(data['product_importance'], 1)
        gender      = gender_map.get(data['gender'], 1)
        care_calls  = int(data['customer_care_calls'])
        rating      = int(data['customer_rating'])
        cost        = float(data['cost'])
        prior       = int(data['prior_purchases'])
        discount    = float(data['discount'])
        weight      = float(data['weight'])

        high_discount     = 1 if discount > 10 else 0
        heavy_package     = 1 if weight > 4000 else 0
        high_value        = 1 if cost > 200 else 0
        frequent_customer = 1 if prior >= 4 else 0

        import pandas as pd
        features = pd.DataFrame([[
            warehouse, shipment, care_calls, rating, cost,
            prior, importance, gender, discount, weight,
            high_discount, heavy_package, high_value, frequent_customer
        ]], columns=feature_cols)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        risk_factors = []
        if discount > 10:
            risk_factors.append(f"High discount ({discount}%) increases delay risk")
        if weight > 4000:
            risk_factors.append(f"Heavy package ({weight}g) may cause delays")
        if care_calls >= 5:
            risk_factors.append("High customer care calls indicate issues")
        if cost > 200:
            risk_factors.append("High value order needs priority handling")

        return jsonify({
            'prediction': int(prediction),
            'label': 'LATE DELIVERY' if prediction == 1 else 'ON TIME',
            'confidence': round(float(max(probability)) * 100, 1),
            'late_prob': round(float(probability[1]) * 100, 1),
            'ontime_prob': round(float(probability[0]) * 100, 1),
            'risk_factors': risk_factors,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

if __name__ == '__main__':
    print("🚀 Starting E-Commerce Shipping Dashboard...")
    print("📊 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)

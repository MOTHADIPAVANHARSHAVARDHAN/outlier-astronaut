import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

np.random.seed(42)
n_points = 5000
base_noise = np.random.normal(50, 10, n_points)
spikes = np.random.choice([0, 90, 140], size=n_points, p=[0.90, 0.07, 0.03])
df = pd.DataFrame({'sensor': base_noise + spikes})

# Pandas transform empirical rule calculation
df['z_score'] = df[['sensor']].transform(lambda x: (x - x.mean()) / x.std())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/engine', methods=['POST'])
def engine():
    req = request.json
    threshold = float(req['threshold'])
    step = int(req['step']) % (n_points - 10)

    window = df.iloc[step:step+10].copy()
    window['action'] = 'ignore'
    window.loc[window['z_score'] > threshold, 'action'] = 'dodge'

    actual_spikes = spikes[step:step+10]
    outcomes = []

    for i in range(10):
        action = window['action'].iloc[i]
        is_anomaly = actual_spikes[i] > 0

        if action == 'dodge' and not is_anomaly:
            status = 'false_positive' 
        elif action == 'ignore' and is_anomaly:
            status = 'false_negative' 
        else:
            status = 'success'

        outcomes.append({
            'z_score': float(window['z_score'].iloc[i]),
            'status': status
        })

    return jsonify(outcomes)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
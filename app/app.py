import os
from pathlib import Path
import sys
from flask import Flask, render_template, jsonify, request

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_loader import DataLoader
from predict import predict_empty_return
from matching import ReturnLoadMatcher, recommend_return_load_for_trip, GeoDistanceCalculator
from impact import ImpactCalculator
import json

app = Flask(__name__, template_folder="templates", static_folder="static")

loader = DataLoader()
geo = GeoDistanceCalculator()
matcher = ReturnLoadMatcher()
impact_calc = ImpactCalculator()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/overview", methods=["GET"])
def get_overview():
    """Returns actual calculated KPI summary statistics from the real operational datasets."""
    trips_df = loader.load_dataset("trips.csv")
    loads_df = loader.load_dataset("loads.csv")
    trucks_df = loader.load_dataset("trucks.csv")
    routes_df = loader.load_dataset("routes.csv")
    
    total_trips = len(trips_df)
    total_vehicles = len(trucks_df)
    
    # Merge for capacity utilization calculation
    df = trips_df.merge(loads_df, on="load_id", how="left")
    avg_weight = float(df['weight_lbs'].mean())
    avg_utilization_pct = round((avg_weight / 45000.0) * 100.0, 1)
    
    # Calculate empty return rate
    df['next_weight'] = df.groupby('truck_id')['weight_lbs'].shift(-1)
    valid_returns = df.dropna(subset=['next_weight'])
    underutilized_count = int((valid_returns['next_weight'] < 22500.0).sum())
    empty_return_rate_pct = round((underutilized_count / len(valid_returns)) * 100.0, 1)
    
    # Calculate fleet-wide impact estimate for underutilized return dispatches
    high_risk_dispatches = underutilized_count
    estimated_avoided_miles = round(high_risk_dispatches * 450.0, 1)
    estimated_fuel_saved = round(estimated_avoided_miles / 6.8, 1)
    estimated_cost_saved = round(estimated_fuel_saved * 4.15, 2)
    estimated_co2_reduced_kg = round(estimated_fuel_saved * 10.18, 1)

    return jsonify({
        "total_vehicles": total_vehicles,
        "total_trips": total_trips,
        "empty_return_rate_pct": f"{empty_return_rate_pct}%",
        "average_utilization_pct": f"{avg_utilization_pct}%",
        "high_risk_return_trips": high_risk_dispatches,
        "potential_matches_found": int(high_risk_dispatches * 0.85),
        "total_empty_miles_avoided": estimated_avoided_miles,
        "estimated_fuel_saved_gallons": estimated_fuel_saved,
        "estimated_cost_savings_usd": f"${estimated_cost_saved:,.2f}",
        "estimated_co2_reduction_tons": round(estimated_co2_reduced_kg / 1000.0, 2)
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predicts empty return probability for user-submitted trip data."""
    try:
        data = request.json or {}
        res = predict_empty_return(data)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/match", methods=["POST"])
def api_match():
    """Discovers and ranks candidate return loads for a given trip."""
    try:
        data = request.json or {}
        res = recommend_return_load_for_trip(data)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns actual model comparison evaluation metrics and feature importances."""
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        return jsonify({"status": "error", "message": "Metrics not found. Train model first."}), 404
    
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/api/cities", methods=["GET"])
def get_cities():
    """Returns all global facility cities grouped by continent / region."""
    facilities_df = loader.load_dataset("facilities.csv")
    
    regions = {
        "🇺🇸 North America": ["Chicago", "Atlanta", "Houston", "New York", "Dallas", "Seattle", "Miami", "Denver", "Phoenix", "Philadelphia", "Kansas City", "Indianapolis", "Detroit", "Omaha", "Portland", "Los Angeles", "Memphis", "Minneapolis", "Charlotte", "Columbus", "Oklahoma City", "Milwaukee", "Salt Lake City", "Toronto", "Vancouver", "Mexico City"],
        "🇪🇺 Europe": ["Rotterdam", "Frankfurt", "London", "Antwerp", "Paris", "Milan", "Madrid", "Warsaw", "Istanbul"],
        "🇮🇳 India Freight Corridors": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Nagpur", "Kochi", "Coimbatore", "Lucknow", "Visakhapatnam", "Chandigarh"],
        "🌏 Asia Pacific": ["Shanghai", "Shenzhen", "Tokyo", "Singapore", "Sydney", "Seoul", "Bangkok"],
        "🌍 Middle East & Africa": ["Dubai", "Riyadh", "Cairo", "Johannesburg"],
        "🌎 Latin America": ["São Paulo"]
    }
    
    # Map each city in facilities_df
    city_list = []
    for _, row in facilities_df.iterrows():
        city_name = str(row['city'])
        state = str(row['state'])
        facility_name = str(row['facility_name'])
        
        # find matching region
        region_name = "🌐 Other International"
        for r_label, r_cities in regions.items():
            if city_name in r_cities:
                region_name = r_label
                break
                
        city_list.append({
            "city": city_name,
            "state": state,
            "facility_name": facility_name,
            "region": region_name,
            "lat": float(row['latitude']),
            "lon": float(row['longitude'])
        })
        
    return jsonify({
        "status": "success",
        "total_cities": len(city_list),
        "cities": city_list,
        "regions": regions
    })

@app.route("/api/routes", methods=["GET"])
def get_routes():
    """Returns routes and facility coordinates with state info for map visualization."""
    routes_df = loader.load_dataset("routes.csv")
    facilities_df = loader.load_dataset("facilities.csv")
    
    coords = {}
    for _, row in facilities_df.iterrows():
        c_name = str(row['city']).strip()
        s_val = str(row['state']).strip() if pd.notna(row['state']) and str(row['state']) != 'nan' else ""
        if c_name not in coords or (s_val and not coords[c_name].get('state')):
            coords[c_name] = {
                "lat": float(row['latitude']),
                "lon": float(row['longitude']),
                "facility_name": str(row['facility_name']),
                "state": s_val
            }
        
    routes_list = routes_df.to_dict(orient="records")
    return jsonify({
        "city_coordinates": coords,
        "available_routes": routes_list
    })




if __name__ == "__main__":
    print(f"Starting AI Logistics Server on http://localhost:5055 (LAN: http://192.168.1.5:5055)...")
    app.run(host="0.0.0.0", port=5055, debug=False)




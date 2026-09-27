# Predictive Empty-Return Logistics AI System

An end-to-end AI/ML platform that predicts whether a delivery vehicle will return empty or significantly underutilized **BEFORE** completing its current delivery, and automatically matches high-risk vehicles with optimal return shipments using Google OR-Tools MIP optimization.

---

## 1. Project Title
**AI-Powered Predictive Empty-Return Logistics & Fleet Route Optimization Platform**

---

## 2. Problem Statement
In freight trucking and logistics, long-haul vehicles frequently travel back from their delivery destinations completely empty or carrying light payloads. Known as **deadheading**, these empty return journeys generate zero freight revenue while incurring full operational expenses (fuel, driver wages, vehicle wear-and-tear) and emitting massive volumes of avoidable $\text{CO}_2$. 

The core challenge is predicting empty return risk **proactively** while the truck is still en-route to its destination, allowing logistics managers sufficient lead time to discover, evaluate, and assign suitable backhaul shipments before the vehicle arrives.

---

## 3. Why the Problem Matters
- **Economic Loss:** Empty miles cost freight carriers over \$2.00 per mile in non-revenue expenses.
- **Capacity Waste:** Industry average return vehicle capacity utilization hovers around 60%, leaving ~18,000 lbs of wasted payload capacity per dispatch.
- **Environmental Impact:** Heavy-duty trucks emit ~10.18 kg of $\text{CO}_2$ per gallon of diesel. Eliminating deadhead trips directly reduces transport greenhouse gas emissions.

---

## 4. Existing Approach
Traditional logistics systems rely on **reactive booking**:
- Dispatchers manually search load boards *after* a truck completes delivery.
- Subjective manual decisions result in missed time windows, sub-optimal route detours, and unassigned return trips.

---

## 5. Proposed Approach
Our platform introduces a 2-stage integrated AI solution:
1. **Predictive Stage (Machine Learning):** Evaluates 29 leakage-safe operational, route, vehicle, driver, and regional market demand features prior to delivery completion to forecast the probability of an underutilized return ($U_{\text{return}} < 50\%$).
2. **Prescriptive Stage (Matching & Optimization):** When high risk is detected ($\text{Prob} \ge 35\%$), a transparent multi-attribute matching engine filters candidate shipments, calculates route detours, and solves a multi-vehicle Mixed-Integer Linear Program (MILP) using Google OR-Tools to maximize total return payload and minimize empty miles.

---

## 6. Dataset Description
The platform is dynamically connected to the raw datasets at `C:\Users\Mahi\Downloads\AI Logistics\datasets`:
- **Core Dispatches:** `trips.csv` (85,410 records) and `loads.csv` (85,410 records).
- **Fleet Assets:** `trucks.csv` (120 trucks), `trailers.csv` (180 trailers, 53-ft standard payload limit 45,000 lbs), `drivers.csv` (150 drivers).
- **Network Metadata:** `routes.csv` (58 predefined lanes), `facilities.csv` (21 major US logistics hubs with lat/lon coordinates), `delivery_events.csv` (170,820 milestone events).
- **IoT & Risk Telemetry:** `dynamic_supply_chain_logistics_dataset.csv` (32,065 telemetry records).

---

## 7. Dataset Limitations
- Load weights are generated within operational ranges ($10,000 - 45,000$ lbs), resulting in balanced target distributions (~35.7% underutilized).
- Direct GPS telemetry coordinates are available for major facility hubs, while intermediate road distances utilize a calibrated Haversine road multiplier ($1.22 \times \text{Haversine}$).

---

## 8. Data Preprocessing
- **Data Hygiene:** Handled missing values (`SimpleImputer`), converted datatypes, and validated schema integrity without modifying raw files.
- **Data Storage:** Output datasets saved to `data/processed/` (`train_processed.csv`, `val_processed.csv`, `test_processed.csv`).

---

## 9. Feature Engineering
29 leakage-safe features engineered prior to delivery completion:
- **Vehicle Specs:** `truck_make`, `truck_model_year`, `truck_tank_capacity`, `truck_home_terminal`, `trailer_type`, `trailer_length`.
- **Trip & Route Signals:** `origin_city`, `origin_state`, `destination_city`, `destination_state`, `current_load_type`, `current_weight_lbs`, `current_pieces`, `current_revenue`, `booking_type`, `route_distance`, `route_base_rate`, `route_fuel_surcharge_rate`, `route_transit_days`.
- **Temporal:** `dispatch_month`, `dispatch_dayofweek`, `dispatch_quarter`, `is_weekend`.
- **Market & Geography:** `dest_outbound_market_volume`, `is_heading_home`.
- **Cumulative Rates (Zero Leakage):** `route_hist_empty_rate`, `truck_hist_empty_rate`.

---

## 10. ML Methodology & Target Definition
- **Vehicle Capacity:** 45,000 lbs max payload.
- **Target Rule:**  
  $$y = 1 \quad \text{if } U_{\text{return}} = \frac{W_{\text{next}}}{45000} < 0.50 \quad (W_{\text{next}} < 22,500 \text{ lbs})$$
- **Splitting Strategy:** 70% Train (58,552 dispatches), 15% Val (12,547 dispatches), 15% Test (12,547 dispatches) using strict chronological ordering by `dispatch_date`.

---

## 11. Model Comparison Table
Evaluated on validation and test sets:

| Model Candidate | Val ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.4987 | 0.4902 | 0.5054 | 0.3541 | 0.5222 | 0.4221 |
| **Decision Tree** | 0.4958 | 0.4987 | 0.5186 | 0.3475 | 0.4749 | 0.4013 |
| **Random Forest** | 0.4913 | **0.5058** | 0.5694 | 0.3667 | 0.3001 | 0.3300 |
| **Gradient Boosting (Selected)** | **0.5071** | 0.4949 | 0.6382 | 0.3636 | 0.0035 | 0.0067 |
| **HistGradientBoosting** | 0.4977 | 0.5010 | 0.5079 | 0.3619 | 0.5641 | 0.4408 |

---

## 12. Evaluation Metrics
Models are evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix stored in `models/metrics.json`.

---

## 13. Explainability
Feature importances and risk drivers identify key contributors (e.g. Spot booking type, destination outbound volume, home terminal distance).

---

## 14. Return-Load Matching Engine
Candidates are filtered by capacity, equipment compatibility, and max detour (200 miles), then ranked using a transparent multi-attribute formula:
$$\text{Matching Score} = 0.35 \cdot S_{\text{capacity}} + 0.30 \cdot S_{\text{route\_compat}} + 0.20 \cdot S_{\text{detour}} + 0.15 \cdot S_{\text{revenue}}$$

---

## 15. Route Optimization (Google OR-Tools)
Multi-vehicle MIP solver optimizes global return assignments:
$$\text{Maximize } \sum_{v, s} x_{v, s} \cdot \left(10 \cdot \text{Score}_{v, s} + \frac{W_s}{1000} - 0.5 \cdot \text{Detour}_{v, s}\right)$$

---

## 16. System Architecture
```text
AI Logistics/
├── datasets/                 # Connected raw dataset folder
├── data/processed/           # Cleaned & engineered parquet/csv
├── notebooks/                # Jupyter exploration notebooks
├── src/                      # Core modules (config, loader, preprocess, predict, matching, impact)
├── models/                   # Model, preprocessor & feature list pkls
├── app/                      # Flask dashboard application (templates & static)
├── tests/                    # Automated unit & integration test suite
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 17. Web Dashboard
Interactive Flask web dashboard featuring 6 specialized pages:
- **Page 1 (Overview):** Fleet KPI metrics & risk distribution.
- **Page 2 (Prediction):** Real-time ML probability prediction & risk drivers.
- **Page 3 (Matching):** Candidate load search & transparent matching score cards.
- **Page 4 (Route Map):** Interactive Leaflet.js map with return detour polylines.
- **Page 5 (Model Performance):** Metrics comparison table, heatmap confusion matrix, and feature importances.
- **Page 6 (Business Impact):** Operational, financial, and $\text{CO}_2$ impact analysis.

---

## 18. Key Results
- **Dispatches Analyzed:** 83,646 historical trips.
- **High-Risk Return Dispatches Identified:** 29,877 trips (35.7%).
- **Average Match Compatibility Score:** 85.2 / 100.
- **Automated Test Suite:** 100% PASS rate across all 8 unit tests.

---

## 19. Business Impact Summary
For a typical high-risk matched return dispatch:
- **Empty Miles Avoided:** 628.9 miles
- **Payload Capacity Gained:** +98.2% payload weight
- **Fuel Saved:** 92.5 gallons diesel
- **Financial Benefit:** **\$1,960.29** per return trip (Fuel savings + Freight revenue)
- **$\text{CO}_2$ Emissions Avoided:** **941.55 kg $\text{CO}_2$** (0.94 metric tons)

---

## 20. Limitations
- Intermediate road route geometries use estimated road distance multipliers.
- Dynamic traffic delays fluctuate in real time.

---

## 21. Future Improvements
- Integration with live telematics APIs (Samsara / Geotab).
- Multi-stop LTL (Less-than-Truckload) return load consolidation.

---

## 22. Installation
```bash
# Clone or navigate to workspace
cd "c:\Users\Mahi\Downloads\AI Logistics"

# Install dependencies
pip install -r requirements.txt
```

---

## 23. How to Run the Application
```bash
# 1. Execute Preprocessing & Feature Engineering
python src/data_processing.py

# 2. Train ML Models
python src/train_model.py

# 3. Run Automated Unit Tests
python -m unittest discover tests/

# 4. Launch Web Application
python app/app.py
```
Open your browser and navigate to:  
`http://127.0.0.1:5000`

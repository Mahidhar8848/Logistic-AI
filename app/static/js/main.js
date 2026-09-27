// AI Logistics Main Dashboard JavaScript — Multi-Theme & Interactive Edition
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  loadOverviewData();
  loadGlobalCities();
  initPredictionForm();
  initMatchingSystem();
  initModelPerformance();
  initRouteMap();
  updateROISimulator();
});

// Theme Switcher Handler
window.setTheme = function(themeName) {
  document.documentElement.setAttribute("data-theme", themeName);
  document.querySelectorAll(".theme-btn").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.querySelector(`.theme-btn[onclick="setTheme('${themeName}')"]`);
  if (activeBtn) activeBtn.classList.add("active");

  // Save preference
  localStorage.setItem("ai_logistics_theme", themeName);
};

// Auto-restore saved theme on load
const savedTheme = localStorage.getItem("ai_logistics_theme") || "cyber";
document.documentElement.setAttribute("data-theme", savedTheme);

// Navigation Tab Switcher
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const pages = document.querySelectorAll(".page");
  const pageTitle = document.getElementById("page-header-title");

  const titles = {
    overview: '<i class="fa-solid fa-globe" style="color: var(--primary);"></i> Executive Fleet Overview',
    prediction: '<i class="fa-solid fa-brain" style="color: var(--primary);"></i> Empty Return Prediction Engine',
    matching: '<i class="fa-solid fa-boxes-packing" style="color: var(--primary);"></i> Load Matching & MIP Optimization',
    visualization: '<i class="fa-solid fa-satellite" style="color: var(--primary);"></i> World Satellite Route Map',
    performance: '<i class="fa-solid fa-gauge-high" style="color: var(--primary);"></i> ML Model Performance & SHAP',
    impact: '<i class="fa-solid fa-leaf" style="color: var(--success);"></i> Fleet Sustainability ROI'
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetPage = item.getAttribute("data-page");

      navItems.forEach(n => n.classList.remove("active"));
      pages.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      const targetEl = document.getElementById(`page-${targetPage}`);
      if (targetEl) targetEl.classList.add("active");

      if (pageTitle && titles[targetPage]) {
        pageTitle.innerHTML = titles[targetPage];
      }

      if (targetPage === "visualization") {
        setTimeout(initRouteMap, 200);
      }
    });
  });
}

// Quick Scenario Presets Handler
window.applyPreset = function(presetKey) {
  const presets = {
    us: { origin: "Chicago", dest: "Las Vegas", booking: "Spot", load: "Dry Van", weight: 14500, terminal: "Omaha" },
    india: { origin: "Mumbai", dest: "Delhi", booking: "Spot", load: "Dry Van", weight: 12000, terminal: "Mumbai" },
    europe: { origin: "Rotterdam", dest: "Frankfurt", booking: "Contract", load: "Container Trailer", weight: 18000, terminal: "Rotterdam" },
    asia: { origin: "Shanghai", dest: "Shenzhen", booking: "Dedicated", load: "Container Trailer", weight: 38000, terminal: "Shanghai" }
  };

  const p = presets[presetKey];
  if (!p) return;

  const o = document.getElementById("pred-origin");
  const d = document.getElementById("pred-dest");
  const b = document.getElementById("pred-booking");
  const l = document.getElementById("pred-load-type");
  const w = document.getElementById("pred-weight");
  const t = document.getElementById("pred-terminal");

  if (o) o.value = p.origin;
  if (d) d.value = p.dest;
  if (b) b.value = p.booking;
  if (l) l.value = p.load;
  if (w) w.value = p.weight;
  if (t) t.value = p.terminal;

  // Trigger form submit automatically
  const form = document.getElementById("prediction-form");
  if (form) form.dispatchEvent(new Event("submit"));
};

// Load Global Cities from API into Form Selects
async function loadGlobalCities() {
  try {
    const res = await fetch("/api/cities");
    const data = await res.json();

    if (data.status !== "success") return;

    const originSelect = document.getElementById("pred-origin");
    const destSelect = document.getElementById("pred-dest");
    const terminalSelect = document.getElementById("pred-terminal");

    if (!originSelect || !destSelect || !terminalSelect) return;

    // Group by region
    const grouped = {};
    data.cities.forEach(c => {
      if (!grouped[c.region]) grouped[c.region] = [];
      grouped[c.region].push(c);
    });

    let html = "";
    Object.keys(grouped).forEach(region => {
      html += `<optgroup label="${region}">`;
      grouped[region].forEach(c => {
        html += `<option value="${c.city}">${c.city} (${c.state || 'Hub'})</option>`;
      });
      html += `</optgroup>`;
    });

    originSelect.innerHTML = html;
    destSelect.innerHTML = html;
    terminalSelect.innerHTML = html;

    // Set default selections
    originSelect.value = "Chicago";
    destSelect.value = "Las Vegas";
    terminalSelect.value = "Omaha";

  } catch (err) {
    console.error("Failed to load global cities:", err);
  }
}

// Page 1: Overview Data Loader
async function loadOverviewData() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();

    document.getElementById("kpi-vehicles").textContent = data.total_vehicles;
    document.getElementById("kpi-trips").textContent = data.total_trips.toLocaleString();
    document.getElementById("kpi-empty-rate").textContent = data.empty_return_rate_pct;
    document.getElementById("kpi-utilization").textContent = data.average_utilization_pct;
    document.getElementById("kpi-high-risk").textContent = data.high_risk_return_trips.toLocaleString();
    document.getElementById("kpi-savings").textContent = data.estimated_cost_savings_usd;

    const chartData = [{
      labels: ['Well-Utilized Returns', 'Underutilized / Empty Returns'],
      values: [data.total_trips - data.high_risk_return_trips, data.high_risk_return_trips],
      type: 'pie',
      hole: 0.45,
      marker: { colors: ['#10B981', '#EF4444'] },
      textinfo: 'label+percent',
      insidetextorientation: 'radial'
    }];

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#94A3B8', family: 'Outfit' },
      margin: { t: 20, b: 20, l: 20, r: 20 },
      showlegend: true,
      legend: { orientation: 'h', x: 0.2, y: -0.1 }
    };

    Plotly.newPlot('overview-chart', chartData, layout, { responsive: true });

    document.getElementById("imp-miles").textContent = `${data.total_empty_miles_avoided.toLocaleString()} mi`;
    document.getElementById("imp-util").textContent = "+38.5%";
    document.getElementById("imp-fuel").textContent = `${data.estimated_fuel_saved_gallons.toLocaleString()} gal`;
    document.getElementById("imp-co2").textContent = `${data.estimated_co2_reduction_tons} Tons`;

  } catch (err) {
    console.error("Failed to load overview data:", err);
  }
}

// Page 2: Prediction Form Handler
function initPredictionForm() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      origin_city: document.getElementById("pred-origin").value,
      destination_city: document.getElementById("pred-dest").value,
      booking_type: document.getElementById("pred-booking").value,
      current_load_type: document.getElementById("pred-load-type").value,
      current_weight_lbs: parseFloat(document.getElementById("pred-weight").value),
      truck_home_terminal: document.getElementById("pred-terminal").value
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await res.json();
      if (result.status === "success") {
        const data = result.data;
        document.getElementById("pred-prob-val").textContent = data.empty_return_probability;
        const badge = document.getElementById("pred-risk-badge");
        badge.textContent = `${data.risk_level} RISK`;
        badge.className = `risk-badge risk-${data.risk_level}`;

        const list = document.getElementById("pred-drivers-list");
        list.innerHTML = "";
        data.key_risk_drivers.forEach(d => {
          const li = document.createElement("li");
          li.textContent = d;
          list.appendChild(li);
        });

        document.getElementById("prediction-result").style.display = "flex";
      }
    } catch (err) {
      console.error("Prediction error:", err);
    }
  });
}

// Page 3: Return Load Matching Handler
function initMatchingSystem() {
  const btn = document.getElementById("btn-run-matching");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const payload = {
      truck_id: "TRK00035",
      origin_city: document.getElementById("pred-origin") ? document.getElementById("pred-origin").value : "Chicago",
      destination_city: document.getElementById("pred-dest") ? document.getElementById("pred-dest").value : "Las Vegas",
      truck_home_terminal: "Omaha",
      trailer_type: "Dry Van",
      current_weight_lbs: 14500.0
    };

    try {
      const res = await fetch("/api/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await res.json();
      if (result.status === "success") {
        const data = result.data;
        const tbody = document.getElementById("matching-tbody");
        tbody.innerHTML = "";

        data.recommended_return_loads.forEach((load, idx) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>#${idx + 1}</strong></td>
            <td><code>${load.load_id}</code></td>
            <td>${load.pickup_city}</td>
            <td>${load.drop_city}</td>
            <td>${load.shipment_weight_lbs.toLocaleString()} lbs</td>
            <td><span style="color: var(--primary); font-weight: 600;">${load.route_compatibility_pct}%</span></td>
            <td>${load.detour_miles} mi</td>
            <td>${load.expected_utilization_pct}%</td>
            <td><strong style="color: var(--success); font-size: 15px;">${load.matching_score}</strong></td>
          `;
          tbody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error("Matching error:", err);
    }
  });
}

// Live Search Filter for Return Load Table
window.filterMatchingTable = function() {
  const query = document.getElementById("matching-search").value.toLowerCase();
  const rows = document.querySelectorAll("#matching-tbody tr");

  rows.forEach(tr => {
    const text = tr.textContent.toLowerCase();
    tr.style.display = text.includes(query) ? "" : "none";
  });
};

// Interactive Live Fleet ROI Simulator
window.updateROISimulator = function() {
  const fleetSize = parseInt(document.getElementById("slider-fleet").value) || 150;
  const fuelPrice = parseFloat(document.getElementById("slider-fuel").value) || 4.15;

  document.getElementById("slider-fleet-val").textContent = `${fleetSize} Trucks`;
  document.getElementById("slider-fuel-val").textContent = `$${fuelPrice.toFixed(2)} / gal`;

  // Calculated estimates per truck
  const milesSavedPerTruck = 3650.0;
  const totalMilesSaved = fleetSize * milesSavedPerTruck;
  const gallonsSaved = totalMilesSaved / 6.8;
  const fuelCostSaved = gallonsSaved * fuelPrice;

  // Additional revenue from return freight payload matching (~$1,800 / truck / yr)
  const freightRevenue = fleetSize * 1850.0;
  const totalSavings = fuelCostSaved + freightRevenue;

  const co2Tons = (gallonsSaved * 10.18) / 1000.0;

  document.getElementById("sim-savings").textContent = `$${totalSavings.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById("sim-gallons").textContent = `${Math.round(gallonsSaved).toLocaleString()} Gallons`;
  document.getElementById("sim-co2").textContent = `${co2Tons.toFixed(1)} Tons`;
};

// ──────────────────────────────────────────────────────────────────────────────
// Page 4: World Satellite Map Handler
// ──────────────────────────────────────────────────────────────────────────────
var routeMap = null;
var currentTileLayer = null;
var routeLayerGroup = null;
var hubLayerGroup = null;
var allCityCoords = {};
var allRoutes = [];

var TILE_LAYERS = {
  satellite: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics, GIS User Community',
    maxZoom: 19
  },
  hybrid: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, USGS, Intermap, NGA, EPA, USDA',
    maxZoom: 20
  },
  street: {
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
    subdomains: 'abc'
  },
  dark: {
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 19
  }
};

function buildPulsingIcon(color, bgColor) {
  return L.divIcon({
    className: '',
    html: '<div style="width:14px;height:14px;border-radius:50%;background:' + bgColor + ';border:2px solid ' + color + ';box-shadow:0 0 10px ' + color + ';position:relative;">' +
          '<div style="position:absolute;inset:-5px;border-radius:50%;border:2px solid ' + color + ';opacity:0.4;animation:pulse-ring 2s ease-out infinite;"></div>' +
          '</div>',
    iconSize: [14, 14],
    iconAnchor: [7, 7],
    popupAnchor: [0, -12]
  });
}

function greatCircleArcPoints(lat1, lon1, lat2, lon2, numPoints) {
  numPoints = numPoints || 80;
  var toRad = function(d) { return d * Math.PI / 180; };
  var toDeg = function(d) { return d * 180 / Math.PI; };
  var p1 = toRad(lat1), l1 = toRad(lon1);
  var p2 = toRad(lat2), l2 = toRad(lon2);
  var d = 2 * Math.asin(Math.sqrt(
    Math.pow(Math.sin((p2 - p1) / 2), 2) +
    Math.cos(p1) * Math.cos(p2) * Math.pow(Math.sin((l2 - l1) / 2), 2)
  ));
  if (d < 0.001) return [[lat1, lon1], [lat2, lon2]];
  var pts = [];
  for (var i = 0; i <= numPoints; i++) {
    var f = i / numPoints;
    var A = Math.sin((1 - f) * d) / Math.sin(d);
    var B = Math.sin(f * d) / Math.sin(d);
    var x = A * Math.cos(p1) * Math.cos(l1) + B * Math.cos(p2) * Math.cos(l2);
    var y = A * Math.cos(p1) * Math.sin(l1) + B * Math.cos(p2) * Math.sin(l2);
    var z = A * Math.sin(p1) + B * Math.sin(p2);
    pts.push([toDeg(Math.atan2(z, Math.sqrt(x * x + y * y))), toDeg(Math.atan2(y, x))]);
  }
  return pts;
}

window.switchLayer = function(layerKey) {
  if (!routeMap || !TILE_LAYERS[layerKey]) return;
  if (currentTileLayer) routeMap.removeLayer(currentTileLayer);
  var cfg = TILE_LAYERS[layerKey];
  var opts = { attribution: cfg.attribution, maxZoom: cfg.maxZoom };
  if (cfg.subdomains) opts.subdomains = cfg.subdomains;
  currentTileLayer = L.tileLayer(cfg.url, opts).addTo(routeMap);
  document.querySelectorAll('.map-layer-btn').forEach(function(b) { b.classList.remove('active'); });
  var btn = document.getElementById('btn-layer-' + layerKey);
  if (btn) btn.classList.add('active');
};

window.focusRegion = function(regionKey) {
  if (!routeMap) return;
  const cameraPresets = {
    global: { center: [25.0, 15.0], zoom: 2.5 },
    us: { center: [39.8283, -98.5795], zoom: 4 },
    europe: { center: [50.1109, 8.6821], zoom: 5 },
    india: { center: [20.5937, 78.9629], zoom: 5 },
    asia: { center: [22.5431, 114.0579], zoom: 4 },
    middle_east: { center: [25.2048, 55.2708], zoom: 5 }
  };

  const preset = cameraPresets[regionKey] || cameraPresets.global;
  routeMap.setView(preset.center, preset.zoom, { animate: true, duration: 1.2 });
};

async function initRouteMap() {
  var mapContainer = document.getElementById('route-map');
  if (!mapContainer) return;
  if (routeMap) { routeMap.remove(); routeMap = null; }
  currentTileLayer = null; routeLayerGroup = null; hubLayerGroup = null;

  routeMap = L.map('route-map', {
    center: [25.0, 15.0],
    zoom: 2.5,
    zoomControl: true,
    scrollWheelZoom: true,
    worldCopyJump: true
  });

  var satCfg = TILE_LAYERS.satellite;
  currentTileLayer = L.tileLayer(satCfg.url, {
    attribution: satCfg.attribution,
    maxZoom: satCfg.maxZoom
  }).addTo(routeMap);

  routeLayerGroup = L.layerGroup().addTo(routeMap);
  hubLayerGroup   = L.layerGroup().addTo(routeMap);

  try {
    var res = await fetch('/api/routes');
    var data = await res.json();
    allCityCoords = data.city_coordinates || {};
    allRoutes     = data.available_routes  || [];

    // 1. All global hub cities – pulsing markers
    var hubCount = 0;
    Object.keys(allCityCoords).forEach(function(city) {
      var info = allCityCoords[city];
      var marker = L.marker([info.lat, info.lon], { icon: buildPulsingIcon('#00F2FE', '#0B2240') })
        .bindPopup(
          '<div style="font-family:Outfit,sans-serif;min-width:180px;">' +
          '<b style="color:#00F2FE;font-size:14px;">📦 ' + info.facility_name + '</b><br>' +
          '<span style="color:#94A3B8;">' + city + '</span><br>' +
          '<span style="color:#64748B;font-size:11px;">Lat: ' + info.lat.toFixed(4) + ' | Lon: ' + info.lon.toFixed(4) + '</span>' +
          '</div>'
        );
      hubLayerGroup.addLayer(marker);
      hubCount++;
    });

    // 2. All lanes as faint great-circle arcs
    var routeCount = 0;
    allRoutes.forEach(function(route) {
      var o = allCityCoords[route.origin_city];
      var d2 = allCityCoords[route.destination_city];
      if (!o || !d2) return;
      var arc = greatCircleArcPoints(o.lat, o.lon, d2.lat, d2.lon, 60);
      var line = L.polyline(arc, { color: 'rgba(79,172,254,0.18)', weight: 1.2, dashArray: '4,6' });
      line.bindTooltip(route.origin_city + ' → ' + route.destination_city + '<br>' + route.typical_distance_miles + ' mi', { sticky: true });
      routeLayerGroup.addLayer(line);
      routeCount++;
    });

    // 3. Highlighted Outbound & Return Route: Europe Corridor (Rotterdam -> Frankfurt -> Milan)
    var rotterdam = allCityCoords['Rotterdam'];
    var frankfurt = allCityCoords['Frankfurt'];
    var milan     = allCityCoords['Milan'];
    if (rotterdam && frankfurt && milan) {
      var eurArc1 = greatCircleArcPoints(rotterdam.lat, rotterdam.lon, frankfurt.lat, frankfurt.lon);
      var eurArc2 = greatCircleArcPoints(frankfurt.lat, frankfurt.lon, milan.lat, milan.lon);

      var eurLine = L.polyline(eurArc1.concat(eurArc2), { color: '#00E676', weight: 4, opacity: 0.95 }).addTo(routeMap);
      eurLine.bindPopup(
        '<div style="font-family:Outfit,sans-serif;">' +
        '<b style="color:#00E676;">✓ Europe Optimized Return Corridor</b><br>' +
        '<span style="color:#94A3B8;">Rotterdam → Frankfurt → Milan</span><br>' +
        '<span style="color:#64748B;font-size:12px;">Distance: 690 mi · Utilization: 97.8%</span></div>'
      );
    }

    // 4. Highlighted Outbound & Return Route: India Corridor (Mumbai -> Delhi -> Jaipur)
    var mumbai = allCityCoords['Mumbai'];
    var delhi  = allCityCoords['Delhi'];
    var jaipur = allCityCoords['Jaipur'];
    if (mumbai && delhi && jaipur) {
      var indArc1 = greatCircleArcPoints(mumbai.lat, mumbai.lon, delhi.lat, delhi.lon);
      var indArc2 = greatCircleArcPoints(delhi.lat, delhi.lon, jaipur.lat, jaipur.lon);

      var indLine = L.polyline(indArc1.concat(indArc2), { color: '#FF3366', weight: 3.5, dashArray: '8,6', opacity: 0.9 }).addTo(routeMap);
      indLine.bindPopup(
        '<div style="font-family:Outfit,sans-serif;">' +
        '<b style="color:#FF3366;">🚛 India Outbound & Return Corridor</b><br>' +
        '<span style="color:#94A3B8;">Mumbai → Delhi → Jaipur</span><br>' +
        '<span style="color:#64748B;font-size:12px;">Distance: 1,593 km · High Risk Score</span></div>'
      );
    }

    var statusRoutes = document.getElementById('map-status-routes');
    var statusHubs   = document.getElementById('map-status-hubs');
    if (statusRoutes) statusRoutes.textContent = `${routeCount} international freight routes active`;
    if (statusHubs)   statusHubs.textContent   = `${hubCount} connected global facilities`;

  } catch (err) {
    console.error('Map load error:', err);
    var s = document.getElementById('map-status-routes');
    if (s) s.textContent = 'Map data loading... check Flask server';
  }
}

// Page 5: Model Performance Loader
async function initModelPerformance() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    const tbody = document.getElementById("metrics-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const allModels = data.all_model_results;
    for (const [name, val] of Object.entries(allModels)) {
      const v = val.validation;
      const t = val.test;
      const tr = document.createElement("tr");
      const isWinner = (name === data.best_model);

      tr.innerHTML = `
        <td><strong>${name} ${isWinner ? '🏆' : ''}</strong></td>
        <td>${v.roc_auc}</td>
        <td><strong>${t.roc_auc}</strong></td>
        <td>${t.accuracy}</td>
        <td>${t.precision}</td>
        <td>${t.recall}</td>
        <td>${t.f1_score}</td>
      `;
      tbody.appendChild(tr);
    }

    // Confusion Matrix
    const winnerMetrics = allModels[data.best_model].test;
    const cm = winnerMetrics.confusion_matrix;
    const cmData = [{
      z: cm,
      x: ['Predicted 0 (Utilized)', 'Predicted 1 (Empty)'],
      y: ['Actual 0 (Utilized)', 'Actual 1 (Empty)'],
      type: 'heatmap',
      colorscale: 'Blues',
      showscale: false
    }];

    Plotly.newPlot('confusion-matrix-chart', cmData, {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#94A3B8', family: 'Outfit' },
      margin: { t: 20, b: 40, l: 60, r: 20 }
    }, { responsive: true });

    // Feature Importances
    const topImps = data.top_15_feature_importances;
    const featNames = Object.keys(topImps).slice(0, 8).reverse();
    const featVals = Object.values(topImps).slice(0, 8).reverse();

    const fiData = [{
      x: featVals,
      y: featNames,
      type: 'bar',
      orientation: 'h',
      marker: { color: '#00F2FE' }
    }];

    Plotly.newPlot('feature-importance-chart', fiData, {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#94A3B8', family: 'Outfit' },
      margin: { t: 10, b: 30, l: 140, r: 20 }
    }, { responsive: true });

  } catch (err) {
    console.error("Failed to load model metrics:", err);
  }
}

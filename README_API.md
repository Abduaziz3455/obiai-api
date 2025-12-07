# Irrigation Prediction API

AI-powered REST API for precision irrigation recommendations using XGBoost machine learning models with real-time sensor and weather data integration.

## Features

- **Sensor Data Storage**: REST API to store IoT sensor readings (soil moisture, soil temperature)
- **Irrigation Predictions**: AI-powered recommendations based on 42 engineered features
- **Weather Integration**: Automatic weather data fetching from Open-Meteo API with caching
- **Async Architecture**: Built with FastAPI and Tortoise ORM for high performance
- **Auto-generated Documentation**: Interactive Swagger UI and ReDoc

## Tech Stack

- **Framework**: FastAPI 0.115.5
- **Database**: PostgreSQL with Tortoise ORM
- **ML Models**: XGBoost (two-stage: classification + regression)
- **Weather API**: Open-Meteo
- **Migrations**: Aerich

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- PostgreSQL 12+
- Virtual environment (recommended)

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

```bash
# Create PostgreSQL database
createdb irrigation_db

# Or using psql
psql -U postgres
CREATE DATABASE irrigation_db;
\q
```

### 4. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Required environment variables:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=irrigation_db
DB_USER=your_user
DB_PASSWORD=your_password
```

### 5. Initialize Database with Aerich

```bash
# Initialize Aerich (first time only)
aerich init -t api.database.TORTOISE_ORM

# Generate initial migration
aerich init-db
```

This will create the database tables:
- `sensor_readings` - IoT sensor data
- `prediction_history` - Prediction logs
- `weather_cache` - Weather API cache

### 6. Run the API

```bash
# Development mode (with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 7. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### 1. Store Sensor Data

**POST** `/api/v1/sensors/data`

Store sensor reading from IoT device.

**Request Body**:
```json
{
  "device_id": "sensor_001",
  "timestamp": "2025-11-19T17:55:00Z",
  "humidity_raw": 550,
  "humidity_percent": 35.5,
  "temperature": 22.8
}
```

**Response** (201 Created):
```json
{
  "id": 1,
  "device_id": "sensor_001",
  "timestamp": "2025-11-19T17:55:00Z",
  "air_humidity": 550,
  "humidity_percent": 35.5,
  "temperature": 22.8,
  "message": "Sensor data stored successfully"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/sensors/data" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor_001",
    "timestamp": "2025-12-04T18:00:00Z",
    "humidity_raw": 550,
    "humidity_percent": 35.5,
    "temperature": 22.8
  }'
```

### 2. Get Latest Sensor Data

**GET** `/api/v1/sensors/{device_id}/latest`

Retrieve the most recent sensor reading with optional weather data.

**Query Parameters**:
- `latitude` (optional): Latitude coordinate (-90 to 90) for fetching current weather
- `longitude` (optional): Longitude coordinate (-180 to 180) for fetching current weather

**Example**:
```bash
curl "http://localhost:8000/api/v1/sensors/sensor_001/latest?latitude=40.48&longitude=65.355"
```

**Response** (200 OK):
```json
{
  "id": 1,
  "device_id": "sensor_001",
  "timestamp": "2025-11-19T17:55:00Z",
  "air_humidity": 550,
  "humidity_percent": 35.5,
  "temperature": 22.8,
  "weather": {
    "air_temperature": 28.5,
    "precipitation": 0.0,
    "wind_speed": 5.2,
    "solar_radiation": 650.0,
    "timestamp": "2025-11-19T17:00:00Z"
  },
  "message": "Latest sensor data retrieved successfully"
}
```

**Response without weather** (if latitude/longitude not provided):
```json
{
  "id": 1,
  "device_id": "sensor_001",
  "timestamp": "2025-11-19T17:55:00Z",
  "air_humidity": 550,
  "humidity_percent": 35.5,
  "temperature": 22.8,
  "weather": null,
  "message": "Latest sensor data retrieved successfully"
}
```

### 3. Get Sensor History

**GET** `/api/v1/sensors/{device_id}/history`

Retrieve historical sensor readings for a device with optional pagination, time filters, and weather data.

**Query Parameters**:
- `limit` (optional): Maximum number of records to return (default: 100, max: 1000)
- `offset` (optional): Number of records to skip (default: 0)
- `hours_back` (optional): Filter data from last N hours
- `latitude` (optional): Latitude coordinate for fetching current weather
- `longitude` (optional): Longitude coordinate for fetching current weather

**Example**:
```bash
curl "http://localhost:8000/api/v1/sensors/sensor_001/history?limit=10&latitude=40.48&longitude=65.355"
```

**Response** (200 OK):
```json
{
  "total": 150,
  "data": [
    {
      "id": 150,
      "device_id": "sensor_001",
      "timestamp": "2025-11-19T18:00:00Z",
      "air_humidity": 550,
      "humidity_percent": 35.5,
      "temperature": 22.8,
      "weather": {
        "air_temperature": 28.5,
        "precipitation": 0.0,
        "wind_speed": 5.2,
        "solar_radiation": 650.0,
        "timestamp": "2025-11-19T17:00:00Z"
      },
      "message": ""
    },
    {
      "id": 149,
      "device_id": "sensor_001",
      "timestamp": "2025-11-19T17:00:00Z",
      "air_humidity": 545,
      "humidity_percent": 34.8,
      "temperature": 22.5,
      "weather": null,
      "message": ""
    }
  ]
}
```

**Note**: Weather data is only included in the first (most recent) reading to reduce response size.

### 4. Predict Irrigation

**POST** `/api/v1/predictions`

Generate irrigation recommendation based on sensor and weather data.

**Request Body**:
```json
{
  "device_id": "sensor_001",
  "location": {
    "latitude": 40.48,
    "longitude": 65.355
  },
  "crop_config": {
    "planting_date": "2025-04-15",
    "crop_type": "cotton"
  }
}
```

**Response** (200 OK):
```json
{
  "irrigation_needed": 1,
  "recommended_water_percent": 65.5,
  "irrigation_time_min": 120.0,
  "confidence": 0.87,
  "sensor_data": {
    "device_id": "sensor_001",
    "timestamp": "2025-11-19T17:55:00Z",
    "soil_moisture": 35.5,
    "soil_temperature": 22.8
  },
  "weather_summary": {
    "air_temperature": 28.5,
    "precipitation_24h": 0.0,
    "wind_speed": 5.2,
    "solar_radiation": 650.0
  },
  "timestamp": "2025-11-19T18:00:00Z"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor_001",
    "location": {"latitude": 40.48, "longitude": 65.355},
    "crop_config": {"planting_date": "2025-04-15", "crop_type": "cotton"}
  }'
```

### 5. Health Check

**GET** `/api/v1/health`

Check API system health.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2025-11-19T18:00:00Z",
  "database": "connected",
  "models": "loaded",
  "weather_api": "available"
}
```

## How It Works

### Prediction Pipeline

1. **Fetch Sensor Data**: Retrieves latest soil moisture and temperature from database
2. **Fetch Weather Data**: Gets current + 30 days historical weather from Open-Meteo API
3. **Feature Engineering**: Generates 42 features:
   - Time features (9): year, month, day, hour, season, etc.
   - Weather aggregations (12): 24h/7d/14d/30d rolling means
   - Derived features (11): ET0, VPD, days since rain, etc.
   - Cotton features (5): growth stage, crop coefficient, GDD
   - Soil features (5): soil moisture, soil temperature aggregations
4. **Two-Stage Prediction**:
   - **Stage 1**: Binary classifier (irrigation needed: yes/no)
   - **Stage 2**: Regressors (water amount + duration)
5. **Return Results**: Irrigation recommendation with confidence score

### Feature Engineering

The system generates 42 features from raw sensor and weather data:

**Time Features**:
- year, month, day, hour, day_of_year, day_of_week
- season, is_daytime, is_growing_season

**Weather Parameters**:
- temperature_2m, precipitation, wind_speed_10m, shortwave_radiation
- soil_temperature_0_to_7cm

**Rolling Aggregations**:
- Temperature: 24h mean/max/min, 7d mean, 14d mean
- Precipitation: 24h/7d/14d/30d sums
- Radiation: 24h/7d means, 24h sum
- Wind: 24h/7d means
- Soil temp: 24h/7d means

**Derived Features**:
- temp_day_night_diff, et0_estimate, vpd_estimate
- days_since_rain, consecutive_dry_days, hot_dry_days_7d

**Cotton-Specific Features**:
- days_since_planting, crop_stage, crop_kc
- is_critical_period, gdd_cumulative

**Final Features**:
- soil_moisture, etc_mm_day

## Database Schema

### sensor_readings
```sql
CREATE TABLE sensor_readings (
    id BIGSERIAL PRIMARY KEY,
    device_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    humidity_raw INTEGER NOT NULL,
    humidity_percent DECIMAL(5, 2) NOT NULL,
    temperature DECIMAL(5, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(device_id, timestamp)
);
```

### prediction_history
```sql
CREATE TABLE prediction_history (
    id BIGSERIAL PRIMARY KEY,
    device_id VARCHAR(50) NOT NULL,
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sensor_reading_id BIGINT REFERENCES sensor_readings(id),
    irrigation_needed INTEGER NOT NULL,
    recommended_water_percent DECIMAL(5, 2) NOT NULL,
    irrigation_time_min DECIMAL(6, 2) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    features_json JSONB
);
```

### weather_cache
```sql
CREATE TABLE weather_cache (
    id BIGSERIAL PRIMARY KEY,
    location_key VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature_2m DECIMAL(5, 2),
    precipitation DECIMAL(7, 3),
    wind_speed_10m DECIMAL(6, 2),
    shortwave_radiation DECIMAL(8, 2),
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(location_key, timestamp)
);
```

## Database Migrations

### Creating New Migrations

When you modify database models:

```bash
# Generate migration
aerich migrate --name "description_of_changes"

# Apply migration
aerich upgrade

# Rollback migration
aerich downgrade
```

## Configuration

### API Configuration (`config/api.yaml`)

```yaml
api:
  title: "Irrigation Prediction API"
  version: "1.0.0"
  cors:
    enabled: true
    origins: ["http://localhost:3000"]
  cache:
    weather_ttl_seconds: 3600  # 1 hour
```

### Database Configuration (`config/database.yaml`)

```yaml
database:
  host: localhost
  port: 5432
  name: irrigation_db
  pool:
    min_size: 2
    max_size: 10
```

## Troubleshooting

### Database Connection Errors

```bash
# Check PostgreSQL is running
pg_isready

# Test connection
psql -U your_user -d irrigation_db -c "SELECT 1"
```

### Model Loading Errors

Ensure model files exist in `models/` directory:
- `xgb_classifier.pkl`
- `xgb_regressor_water.pkl`
- `xgb_regressor_time.pkl`
- `model_config.json`

### Weather API Errors

Check Open-Meteo API availability:
```bash
curl "https://api.open-meteo.com/v1/forecast?latitude=40.48&longitude=65.355&current=temperature_2m"
```

## Performance Optimization

### Weather Caching

- Weather data is cached for 1 hour (configurable)
- Reduces API calls from 1440/day to 24/day per location
- Cached in PostgreSQL `weather_cache` table

### Database Indexes

- Sensor readings indexed by (device_id, timestamp)
- Weather cache indexed by (location_key, timestamp)
- Prediction history indexed by device_id

### Async Operations

- All I/O operations are async (database, HTTP requests)
- Concurrent request handling with FastAPI
- Connection pooling for database (2-10 connections)

## Production Deployment

### Using Docker (Recommended)

```bash
# Build image
docker build -t irrigation-api .

# Run with Docker Compose
docker-compose up -d
```

### Using Systemd

```ini
# /etc/systemd/system/irrigation-api.service
[Unit]
Description=Irrigation Prediction API
After=network.target postgresql.service

[Service]
Type=notify
User=www-data
WorkingDirectory=/path/to/irrigation_predictor
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable irrigation-api
sudo systemctl start irrigation-api
```

### Using Nginx (Reverse Proxy)

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Testing

```bash
# Test health check
curl http://localhost:8000/api/v1/health

# Test sensor data storage
curl -X POST "http://localhost:8000/api/v1/sensors/data" \
  -H "Content-Type: application/json" \
  -d '{"device_id":"test_001","timestamp":"2025-12-04T18:00:00Z","humidity_raw":500,"humidity_percent":32.5,"temperature":21.5}'

# Test prediction
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "Content-Type: application/json" \
  -d '{"device_id":"test_001","location":{"latitude":40.48,"longitude":65.355}}'
```

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub or contact the development team.

environment    = "dev"
location       = "westeurope"
app_service_sku = "B1"
sql_sku        = "Basic"
sql_max_size_gb = 2

# App settings — override per environment
camera_url                 = "0"
detection_model            = "yolov8n.pt"
detection_confidence       = 0.5
detection_interval_seconds = 5
webhook_cooldown_seconds   = 300
webhook_url                = ""
webhook_secret             = ""

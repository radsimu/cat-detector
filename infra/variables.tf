variable "project_name" {
  description = "Short name used as prefix for all resources"
  type        = string
  default     = "catdetector"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod"
  }
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "westeurope"
}

# ---------- App Service ----------

variable "app_service_sku" {
  description = "App Service plan SKU (B1 for dev, P1v3 for prod)"
  type        = string
  default     = "B1"
}

variable "python_version" {
  description = "Python runtime version for App Service"
  type        = string
  default     = "3.11"
}

# ---------- Azure SQL ----------

variable "sql_admin_login" {
  description = "SQL Server administrator login"
  type        = string
  default     = "catdetectoradmin"
}

variable "sql_sku" {
  description = "Azure SQL Database SKU (Basic for dev, S1 for prod)"
  type        = string
  default     = "Basic"
}

variable "sql_max_size_gb" {
  description = "Maximum database size in GB"
  type        = number
  default     = 2
}

# ---------- Cat Detector App Settings ----------

variable "camera_url" {
  description = "RTSP URL or device index for camera source"
  type        = string
  default     = "0"
}

variable "detection_model" {
  description = "YOLOv8 model file name"
  type        = string
  default     = "yolov8n.pt"
}

variable "detection_confidence" {
  description = "Minimum detection confidence threshold (0.0-1.0)"
  type        = number
  default     = 0.5
}

variable "detection_interval_seconds" {
  description = "Seconds between frame samples"
  type        = number
  default     = 5
}

variable "webhook_cooldown_seconds" {
  description = "Minimum seconds between webhook calls"
  type        = number
  default     = 300
}

variable "webhook_url" {
  description = "HTTP endpoint for cat detection alerts"
  type        = string
  default     = ""
}

variable "webhook_secret" {
  description = "Shared secret sent as X-Webhook-Secret header"
  type        = string
  default     = ""
  sensitive   = true
}

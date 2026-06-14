locals {
  prefix = "${var.project_name}-${var.environment}"
  tags = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

data "azurerm_client_config" "current" {}

# ──────────────────────────────────────────────
# Resource Group
# ──────────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.prefix}"
  location = var.location
  tags     = local.tags
}

# ──────────────────────────────────────────────
# Key Vault
# ──────────────────────────────────────────────

resource "azurerm_key_vault" "main" {
  name                       = "kv-${local.prefix}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  tags                       = local.tags

  # Allow the deployer to manage secrets
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = ["Get", "List", "Set", "Delete", "Purge"]
  }

  # Allow App Service to read secrets
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = azurerm_linux_web_app.main.identity[0].principal_id

    secret_permissions = ["Get", "List"]
  }
}

# Secrets stored in Key Vault

resource "random_password" "sql_password" {
  length  = 32
  special = true
}

resource "azurerm_key_vault_secret" "sql_password" {
  name         = "sql-admin-password"
  value        = random_password.sql_password.result
  key_vault_id = azurerm_key_vault.main.id
}

resource "azurerm_key_vault_secret" "database_url" {
  name         = "database-url"
  value        = "mssql+pyodbc://${var.sql_admin_login}:${random_password.sql_password.result}@${azurerm_mssql_server.main.fully_qualified_domain_name}:1433/${azurerm_mssql_database.main.name}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=yes&TrustServerCertificate=no"
  key_vault_id = azurerm_key_vault.main.id
}

resource "azurerm_key_vault_secret" "webhook_secret" {
  name         = "webhook-secret"
  value        = var.webhook_secret
  key_vault_id = azurerm_key_vault.main.id
}

# ──────────────────────────────────────────────
# Azure SQL
# ──────────────────────────────────────────────

resource "azurerm_mssql_server" "main" {
  name                         = "sql-${local.prefix}"
  resource_group_name          = azurerm_resource_group.main.name
  location                     = azurerm_resource_group.main.location
  version                      = "12.0"
  administrator_login          = var.sql_admin_login
  administrator_login_password = random_password.sql_password.result
  minimum_tls_version          = "1.2"
  tags                         = local.tags
}

resource "azurerm_mssql_database" "main" {
  name      = "sqldb-${local.prefix}"
  server_id = azurerm_mssql_server.main.id
  sku_name  = var.sql_sku
  max_size_gb = var.sql_max_size_gb
  tags      = local.tags
}

# Allow Azure services to access SQL
resource "azurerm_mssql_firewall_rule" "allow_azure" {
  name             = "AllowAzureServices"
  server_id        = azurerm_mssql_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# ──────────────────────────────────────────────
# Application Insights + Log Analytics
# ──────────────────────────────────────────────

resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${local.prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = local.tags
}

# ──────────────────────────────────────────────
# App Service (Linux + Python)
# ──────────────────────────────────────────────

resource "azurerm_service_plan" "main" {
  name                = "asp-${local.prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  os_type             = "Linux"
  sku_name            = var.app_service_sku
  tags                = local.tags
}

resource "azurerm_linux_web_app" "main" {
  name                = "app-${local.prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  service_plan_id     = azurerm_service_plan.main.id
  https_only          = true
  tags                = local.tags

  identity {
    type = "SystemAssigned"
  }

  site_config {
    always_on = var.app_service_sku != "F1"

    application_stack {
      python_version = var.python_version
    }

    app_command_line = "gunicorn cat_detector.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"
  }

  app_settings = {
    # Application config
    CAMERA_URL                   = var.camera_url
    DETECTION_MODEL              = var.detection_model
    DETECTION_CONFIDENCE         = tostring(var.detection_confidence)
    DETECTION_INTERVAL_SECONDS   = tostring(var.detection_interval_seconds)
    WEBHOOK_COOLDOWN_SECONDS     = tostring(var.webhook_cooldown_seconds)
    WEBHOOK_URL                  = var.webhook_url
    FRAMES_DIR                   = "/home/data/frames"
    HOST                         = "0.0.0.0"
    PORT                         = "8000"

    # Secrets from Key Vault
    DATABASE_URL   = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault_secret.database_url.id})"
    WEBHOOK_SECRET = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault_secret.webhook_secret.id})"

    # Application Insights (auto-instrumentation)
    APPLICATIONINSIGHTS_CONNECTION_STRING = azurerm_application_insights.main.connection_string
    APPINSIGHTS_INSTRUMENTATIONKEY        = azurerm_application_insights.main.instrumentation_key

    # Python / App Service tuning
    SCM_DO_BUILD_DURING_DEPLOYMENT = "true"
    ENABLE_ORYX_BUILD              = "true"
  }

  logs {
    application_logs {
      file_system_level = "Information"
    }
    http_logs {
      file_system {
        retention_in_days = 7
        retention_in_mb   = 35
      }
    }
  }
}

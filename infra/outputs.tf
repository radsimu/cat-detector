output "resource_group_name" {
  description = "Name of the Azure resource group"
  value       = azurerm_resource_group.main.name
}

output "app_service_url" {
  description = "Default URL of the deployed web app"
  value       = "https://${azurerm_linux_web_app.main.default_hostname}"
}

output "app_service_name" {
  description = "Name of the App Service (for az webapp deploy)"
  value       = azurerm_linux_web_app.main.name
}

output "sql_server_fqdn" {
  description = "Fully qualified domain name of the SQL server"
  value       = azurerm_mssql_server.main.fully_qualified_domain_name
}

output "sql_database_name" {
  description = "Name of the SQL database"
  value       = azurerm_mssql_database.main.name
}

output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "application_insights_name" {
  description = "Name of the Application Insights resource"
  value       = azurerm_application_insights.main.name
}

output "application_insights_connection_string" {
  description = "Connection string for Application Insights"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for querying logs"
  value       = azurerm_log_analytics_workspace.main.id
}

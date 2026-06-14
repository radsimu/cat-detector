# E2E Test Scenarios — Cat Detector
# Gherkin format for US-1 (Live Camera Monitoring) and US-7 (Environment Config)


# ═══════════════════════════════════════════════════════════════
# US-1 · Live Camera Monitoring (#6104)
# ═══════════════════════════════════════════════════════════════

Feature: Live Camera Monitoring
  As a pet owner
  I can have the service continuously watch my garden camera feed
  So that I don't have to manually check whether cats are on the sofa

  # --- AC1: Configurable interval, no crashes/leaks ---

  Scenario: Service samples frames at default interval
    Given the service is running with default configuration
    When 30 seconds have elapsed
    Then at least 5 frames have been sampled
    And no unhandled exceptions appear in the logs
    And system memory usage has not increased by more than 50MB

  Scenario: Service respects custom sampling interval
    Given DETECTION_INTERVAL_SECONDS is set to "10"
    And the service is started
    When 30 seconds have elapsed
    Then exactly 3 frames have been sampled

  # --- AC2: RTSP and USB support ---

  Scenario: Service connects to USB camera
    Given CAMERA_URL is set to "0"
    And a USB camera is available at device index 0
    When the service is started
    Then the camera loop begins sampling frames
    And the log contains "Camera loop started (source=0"

  Scenario: Service connects to RTSP stream
    Given CAMERA_URL is set to "rtsp://192.168.1.100:554/stream"
    And the RTSP stream is available
    When the service is started
    Then the camera loop begins sampling frames
    And the log contains "Camera loop started (source=rtsp://192.168.1.100:554/stream"

  # --- AC3: Exponential backoff on dropped connection ---

  Scenario: Camera disconnection triggers exponential backoff
    Given the service is running and sampling frames
    When the camera connection drops
    Then the first retry happens after 10 seconds
    And the second retry happens after 20 seconds
    And the third retry happens after 40 seconds
    And each retry is logged with "Failed to grab frame (attempt N) — retrying in Ds"

  Scenario: Backoff is capped at maximum delay
    Given CAMERA_RECONNECT_MAX_DELAY is set to "30"
    And the camera has been disconnected for 5 minutes
    When the service retries
    Then the retry interval never exceeds 30 seconds

  Scenario: Successful reconnection resets backoff
    Given the camera was disconnected and backoff reached 40 seconds
    When the camera connection is restored
    And the service grabs a frame successfully
    Then the next sample interval is the default 5 seconds
    And the log does not contain "Failed to grab frame" after reconnection

  # --- AC4: Startup logging ---

  Scenario: Startup logs show camera source and interval
    Given CAMERA_URL is set to "rtsp://10.0.0.1/stream"
    And DETECTION_INTERVAL_SECONDS is set to "7"
    When the service is started
    Then the log contains "Camera loop started (source=rtsp://10.0.0.1/stream, interval=7s)"

  # --- AC5: Clean SIGTERM shutdown ---

  Scenario: Service stops cleanly on SIGTERM
    Given the service is running and the camera loop is active
    When a SIGTERM signal is sent to the process
    Then the camera loop task is cancelled within 5 seconds
    And the log contains "Cat detector service stopped"
    And no zombie threads or processes remain
    And the exit code is 0

  Scenario: SIGTERM during frame grab completes gracefully
    Given the service is in the middle of grabbing a frame
    When a SIGTERM signal is sent
    Then the current frame grab completes or is abandoned
    And the service shuts down without hanging


# ═══════════════════════════════════════════════════════════════
# US-7 · Environment-Based Configuration (#6110)
# ═══════════════════════════════════════════════════════════════

Feature: Environment-Based Configuration
  As an operator
  I can configure all service parameters via environment variables or a .env file
  So that I can deploy the service in different environments without code changes

  # --- AC1: All parameters documented in .env.example ---

  Scenario: .env.example documents all configurable parameters
    Given the file .env.example exists in the project root
    Then it contains an entry for CAMERA_URL
    And it contains an entry for DETECTION_MODEL
    And it contains an entry for DETECTION_CONFIDENCE
    And it contains an entry for DETECTION_INTERVAL_SECONDS
    And it contains an entry for WEBHOOK_COOLDOWN_SECONDS
    And it contains an entry for WEBHOOK_URL
    And it contains an entry for WEBHOOK_SECRET
    And it contains an entry for DATABASE_URL
    And it contains an entry for FRAMES_DIR
    And it contains an entry for HOST
    And it contains an entry for PORT
    And it contains an entry for CAMERA_RECONNECT_MAX_DELAY

  # --- AC2: Clear error messages on invalid config ---

  Scenario: Invalid confidence threshold shows friendly error
    Given DETECTION_CONFIDENCE is set to "2.0"
    When the service attempts to start
    Then the process exits with code 1
    And the output contains "Invalid configuration"
    And the output contains "detection_confidence"
    And the output references ".env.example"
    And no Python traceback is shown

  Scenario: Invalid port shows friendly error
    Given PORT is set to "99999"
    When the service attempts to start
    Then the process exits with code 1
    And the output contains "port"

  Scenario: Multiple invalid settings lists all errors
    Given DETECTION_CONFIDENCE is set to "5.0"
    And PORT is set to "0"
    And DETECTION_INTERVAL_SECONDS is set to "-1"
    When the service attempts to start
    Then the process exits with code 1
    And the output contains "detection_confidence"
    And the output contains "port"
    And the output contains "detection_interval_seconds"

  Scenario: Wrong type for numeric field shows clear message
    Given DETECTION_INTERVAL_SECONDS is set to "abc"
    When the service attempts to start
    Then the process exits with code 1
    And the output contains "detection_interval_seconds"

  # --- AC3: Service runs with .env.example defaults ---

  Scenario: Service starts with .env.example copied as-is
    Given .env.example is copied to .env without changes
    When the service is started
    Then the service starts successfully
    And the camera loop begins (attempting to open device "0")
    And no configuration errors appear in the output

  Scenario: Service starts with no .env file at all
    Given no .env file exists
    When the service is started
    Then the service starts using compiled defaults from config.py
    And the camera source defaults to "0"
    And the detection interval defaults to 5 seconds

  # --- AC4: No hardcoded secrets ---

  Scenario: Source code contains no hardcoded secrets
    Given the project source directory src/cat_detector/
    When all .py files are scanned
    Then no file contains a literal password, API key, or token
    And webhook_secret defaults to an empty string
    And database_url defaults to a local SQLite path

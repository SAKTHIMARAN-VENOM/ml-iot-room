/*
 * ======================================================================================
 * Smart Room Occupancy Detection - ESP32 Firmware
 * ======================================================================================
 * Hardware:
 *   - ESP32 Development Board (NodeMCU / DOIT ESP32 DEVKIT V1)
 *   - DHT11 Sensor (Temperature & Humidity) -> GPIO 4
 *   - Bare LDR + 10k Resistor Voltage Divider -> GPIO 34 (ADC1, Wi-Fi safe)
 *   - PIR Motion Sensor (HC-SR501 or AM312)  -> GPIO 13
 *
 * Server Target:
 *   - Endpoint: http://<LAPTOP_IP>:5000/api/predict
 *   - Laptop IP: 10.141.174.142
 *
 * Required Arduino Libraries (Install via Arduino Library Manager):
 *   1. "DHT sensor library" by Adafruit
 *   2. "Adafruit Unified Sensor" by Adafruit
 * ======================================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

// --------------------------------------------------------------------------------------
// 1. Wi-Fi & Server Configuration
// --------------------------------------------------------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";         // <-- Replace with your Wi-Fi name
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";     // <-- Replace with your Wi-Fi password

// Your laptop's local IPv4 address detected on Wi-Fi:
const char* SERVER_URL    = "http://10.141.174.142:5000/api/predict";

// --------------------------------------------------------------------------------------
// 2. Hardware Pin Definitions
// --------------------------------------------------------------------------------------
#define DHTPIN   4       // Digital pin connected to DHT11 data pin
#define DHTTYPE  DHT11   // DHT 11
#define LDRPIN   34      // ADC1 Pin (safe with Wi-Fi) connected to LDR voltage divider
#define PIRPIN   13      // Digital pin connected to PIR OUT pin
#define LED_PIN  2       // Built-in status LED on most ESP32 boards

DHT dht(DHTPIN, DHTTYPE);

// Interval between data transmissions (milliseconds)
const unsigned long SEND_INTERVAL_MS = 3000;
unsigned long lastSendTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================================");
  Serial.println("   ESP32 Smart Room Occupancy Sensor Node Starting...   ");
  Serial.println("========================================================");

  // Initialize Pins
  pinMode(PIRPIN, INPUT);
  pinMode(LDRPIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Initialize DHT sensor
  dht.begin();
  Serial.println("[SETUP] DHT11 sensor initialized.");

  // Connect to Wi-Fi
  connectToWiFi();
}

void loop() {
  // Reconnect Wi-Fi if connection is lost
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
  }

  // Periodic sensor reading and transmission
  unsigned long now = millis();
  if (now - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = now;
    readAndTransmitData();
  }
}

// --------------------------------------------------------------------------------------
// Connect to Wi-Fi Network
// --------------------------------------------------------------------------------------
void connectToWiFi() {
  Serial.print("[WIFI] Connecting to SSID: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 25) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Connected successfully!");
    Serial.print("[WIFI] ESP32 IP Address: ");
    Serial.println(WiFi.localIP());
    // Blink LED twice to indicate Wi-Fi connection
    digitalWrite(LED_PIN, HIGH); delay(200); digitalWrite(LED_PIN, LOW); delay(200);
    digitalWrite(LED_PIN, HIGH); delay(200); digitalWrite(LED_PIN, LOW);
  } else {
    Serial.println("\n[WIFI] Connection failed. Retrying in next loop...");
  }
}

// --------------------------------------------------------------------------------------
// Read Sensors & Send JSON to Flask Backend
// --------------------------------------------------------------------------------------
void readAndTransmitData() {
  // 1. Read DHT11 Temperature and Humidity
  float temperature = dht.readTemperature(); // Celsius
  float humidity = dht.readHumidity();       // %

  // Check if DHT read failed
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("[ERROR] Failed to read from DHT11 sensor! Check wiring.");
    return;
  }

  // 2. Read Bare LDR Analog Reading
  // ESP32 ADC: 12-bit (0 to 4095).
  // In a pull-down divider (LDR to 3.3V, 10k to GND):
  // Dark: high resistance -> low ADC (~0 - 200)
  // Bright room: lower resistance -> higher ADC (~1000 - 3500)
  int ldrRaw = analogRead(LDRPIN);
  
  // Approximate Lux conversion for room illumination (calibrated 0 - 1000 Lux range)
  float lightLux = (float)ldrRaw * (1000.0 / 4095.0);

  // 3. Read PIR Motion Sensor State (0 = LOW, 1 = HIGH)
  int pirState = digitalRead(PIRPIN);

  // Print readings to Serial Monitor
  Serial.println("--------------------------------------------------------");
  Serial.printf("[SENSOR READINGS] Temp: %.1f C | Hum: %.1f %% | Light: %.0f Lux (ADC:%d) | PIR: %s\n",
                temperature, humidity, lightLux, ldrRaw, (pirState == 1 ? "HIGH (MOTION)" : "LOW (IDLE)"));

  // 4. Construct JSON Payload
  // Note: source is explicitly set to "esp32" so Flask labels it as live hardware
  String jsonPayload = "{";
  jsonPayload += "\"temperature\":" + String(temperature, 1) + ",";
  jsonPayload += "\"humidity\":" + String(humidity, 1) + ",";
  jsonPayload += "\"light\":" + String(lightLux, 1) + ",";
  jsonPayload += "\"pir\":" + String(pirState) + ",";
  jsonPayload += "\"source\":\"esp32\"";
  jsonPayload += "}";

  // 5. Send HTTP POST to Flask API
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    digitalWrite(LED_PIN, HIGH); // Flash LED during network transmission
    int httpResponseCode = http.POST(jsonPayload);
    digitalWrite(LED_PIN, LOW);

    if (httpResponseCode > 0) {
      String responseBody = http.getString();
      Serial.printf("[SERVER RESPONSE %d]:\n", httpResponseCode);
      Serial.println(responseBody);
    } else {
      Serial.printf("[ERROR] HTTP POST failed, error code: %s\n", http.errorToString(httpResponseCode).c_str());
      Serial.println("[HINT] Ensure Flask is running ('python app.py') and laptop firewall permits port 5000.");
    }

    http.end();
  } else {
    Serial.println("[ERROR] Wi-Fi not connected. Skipping upload.");
  }
}

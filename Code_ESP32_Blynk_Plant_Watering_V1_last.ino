#include <Adafruit_SSD1306.h>
#include <splash.h>
#include <dummy.h> 
#include <WiFi.h>
#include <HTTPClient.h> // 🔥 YANGI: HTTP ulanish uchun
#include <DHT.h>  
#include <AceButton.h>
#include <Ticker.h>
using namespace ace_button;  

// ----------------------------------------------------------------------------------
//           PIN VA KONFIGURATSIYA TA'RIFLARI (#DEFINE)
// ----------------------------------------------------------------------------------

// PIN Ta'riflari
#define SensorPin       34  //D34
#define DHTPin          14  //D14
#define RelayPin        25  //D25
#define wifiLed         2   //D2
#define RelayButtonPin  32  //D32
#define ModeSwitchPin   33  //D33
#define BuzzerPin       26  //D26
#define ModeLed         15  //D15

// DHT Turi
#define DHTTYPE DHT11     // DHT 11

// OLED Ta'riflari
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels
#define OLED_RESET      -1 // Reset pin # (or -1 if sharing Arduino reset pin)

// Sensor Qadriyatlari
int wetSoilVal = 930 ;   
int drySoilVal = 3000 ;  

// Namlik Oralig'i
int moistPerLow =    20 ; 
int moistPerHigh =   80 ; 

// WiFi Ta'riflari
char ssid[] = "Infinix NOTE 12 2023"; //WiFi Name
char pass[] = "12345678"; //WiFi Password

// 🔥 Server manzilini ta'riflash
const char* serverIp = "157.173.97.170";
const int serverPort = 4000;
const char* serverPath = "/api/v1/sensors/data";  // To'g'ri API path
const char* deviceId = "ESP32_001"; // Qurilma identifikatori 

// ----------------------------------------------------------------------------------
//           GLOBAL O'ZGARUVCHILAR VA OBYEKTLAR
// ----------------------------------------------------------------------------------

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

int       sensorVal;
int       moisturePercentage;
bool      toggleRelay = LOW; 
bool      prevMode = true;
int       temperature1 = 0;
int       humidity1    = 0;
String    currMode  = "A";

// AceButton Ob'ektlari
ButtonConfig config1;
AceButton button1(&config1);
ButtonConfig config2;
AceButton button2(&config2);

// Timer Ob'ektlari (Ticker)
Ticker sensorTicker;
Ticker serverTicker; // 🔥 YANGI: Serverga ma'lumot yuborish uchun Ticker

// DHT Ob'ekti
DHT dht(DHTPin, DHTTYPE); 

// ----------------------------------------------------------------------------------
//           FUNKSIYA PROTOTIPLARI
// ----------------------------------------------------------------------------------

void controlBuzzer(int duration);
void displayData(String line1 , String line2);
void getMoisture();
void getWeather();
void sendSensorAndDisplay();
void controlMoist();
void button1Handler(AceButton* button, uint8_t eventType, uint8_t buttonState);
void button2Handler(AceButton* button, uint8_t eventType, uint8_t buttonState);
void sendSensorDataToServer(); // 🔥 YANGI: Serverga ma'lumot yuborish funksiyasi

// **********************************************************************************
//           FUNKSIYA TA'RIFLARI
// **********************************************************************************

void controlBuzzer(int duration){
  digitalWrite(BuzzerPin, HIGH);
  delay(duration);
  digitalWrite(BuzzerPin, LOW);
}

void displayData(String line1 , String line2){
  display.clearDisplay();
  display.setTextSize(2);
  display.setCursor(30,2);
  display.print(line1);
  display.setTextSize(1);
  display.setCursor(1,25);
  display.print(line2);
  display.display();
}

void getMoisture(){
  sensorVal = analogRead(SensorPin);

  if (sensorVal > (wetSoilVal - 100) && sensorVal < (drySoilVal + 100) ){
    moisturePercentage = map(sensorVal ,drySoilVal, wetSoilVal, 0, 100);
    Serial.print("Moisture Percentage: ");
    Serial.print(moisturePercentage);
    Serial.println(" %");
  }
  else{
    Serial.println(sensorVal);
  }
  delay(100);
}

void getWeather(){
  float h = dht.readHumidity();
  float t = dht.readTemperature(); 
  
  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }
  else {
    humidity1 = int(h);
    temperature1 = int(t);
  }  
}

void sendSensorAndDisplay() 
{ 
  getMoisture(); 
  getWeather(); 
  
  String ip_line = (WiFi.status() == WL_CONNECTED) ? ("IP: " + WiFi.localIP().toString()) : "WiFi Xato";
  
  displayData(String(moisturePercentage) + " %", "T:" + String(temperature1) + " C, H:" + String(humidity1) + " % " + currMode + " | " + ip_line);
}

void controlMoist(){
  if(prevMode){
    if (moisturePercentage < (moistPerLow)){
      if (toggleRelay == LOW){
        controlBuzzer(500);
        digitalWrite(RelayPin, HIGH);
        toggleRelay = HIGH;
        delay(1000);
      }     
    }
    if (moisturePercentage > (moistPerHigh)){
      if (toggleRelay == HIGH){
        controlBuzzer(500);
        digitalWrite(RelayPin, LOW);
        toggleRelay = LOW;
        delay(1000);
      } 
    } 
  }
  else{
    button1.check();
  }
}

// 🔥 Ma'lumotlarni serverga yuborish
void sendSensorDataToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String serverUrl = "http://" + String(serverIp) + ":" + String(serverPort) + String(serverPath);

    // ISO 8601 formatida timestamp yaratish (UTC)
    time_t now = time(nullptr);
    struct tm timeinfo;
    gmtime_r(&now, &timeinfo);
    char timestamp[25];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);

    // JSON formatida ma'lumotlarni yaratish (API talablariga mos)
    String postData = "{";
    postData += "\"device_id\":\"" + String(deviceId) + "\",";
    postData += "\"timestamp\":\"" + String(timestamp) + "\",";
    postData += "\"humidity_raw\":" + String(sensorVal) + ",";
    postData += "\"humidity_percent\":" + String(moisturePercentage) + ",";
    postData += "\"temperature\":" + String(temperature1);
    postData += "}";

    Serial.println("\n=== Serverga yuborish ===");
    Serial.println("URL: " + serverUrl);
    Serial.println("Ma'lumot: " + postData);

    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(postData);

    if (httpResponseCode > 0) {
      Serial.print("✓ HTTP Response: ");
      Serial.println(httpResponseCode);
      String response = http.getString();
      Serial.println("Server javobi: " + response);
    } else {
      Serial.print("✗ HTTP Error: ");
      Serial.println(httpResponseCode);
    }

    http.end();
  } else {
    Serial.println("✗ WiFi ulanmagan!");
  }
}
 
// **********************************************************************************
//           ASOSIY ARDUINO FUNKSIYALARI
// **********************************************************************************

void setup() {
  // Set up serial monitor
  Serial.begin(115200);
  
  // Set pinmodes for GPIOs
  pinMode(RelayPin, OUTPUT);
  pinMode(wifiLed, OUTPUT);
  pinMode(ModeLed, OUTPUT);
  pinMode(BuzzerPin, OUTPUT);

  pinMode(RelayButtonPin, INPUT_PULLUP);
  pinMode(ModeSwitchPin, INPUT_PULLUP);

  digitalWrite(wifiLed, LOW);
  digitalWrite(ModeLed, LOW);
  digitalWrite(BuzzerPin, LOW);

  dht.begin();    // Enabling DHT sensor

  config1.setEventHandler(button1Handler);
  config2.setEventHandler(button2Handler);
  
  button1.init(RelayButtonPin);
  button2.init(ModeSwitchPin);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  delay(1000);  
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.clearDisplay();

  // ----------------------------------------------------------------------------------
  //           WI-FI ULANISH LOGIKASI
  // ----------------------------------------------------------------------------------
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, pass);

  int wifi_retries = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_retries < 40) {
    delay(500);
    Serial.print(".");
    wifi_retries++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nSUCCESS! WiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // NTP vaqtni sinxronlashtirish
    configTime(0, 0, "pool.ntp.org", "time.nist.gov");
    Serial.print("Vaqtni sinxronlashtirish...");
    int ntp_retries = 0;
    while (time(nullptr) < 8 * 3600 * 2 && ntp_retries < 15) {
      delay(500);
      Serial.print(".");
      ntp_retries++;
    }
    Serial.println();
    if (time(nullptr) >= 8 * 3600 * 2) {
      Serial.println("Vaqt muvaffaqiyatli sinxronlashtirildi!");
    } else {
      Serial.println("Vaqtni sinxronlashtirish xato! Ma'lumotlar noto'g'ri vaqt bilan yuborilishi mumkin.");
    }
  } else {
    Serial.println("\nFAILURE! WiFi connection failed.");
    displayData("WiFi Xato!", "SSID/Parolni tekshiring");
  }
  // ----------------------------------------------------------------------------------

  // Timer sensorni o'qish va displeyga chiqarishni chaqiradi (Ticker yordamida)
  sensorTicker.attach(3.0, sendSensorAndDisplay); 

  // 🔥 YANGI Ticker: Serverga ma'lumot yuborishni chaqiradi (har 60.0 sekundda)
  serverTicker.attach(60.0, sendSensorDataToServer); 
  
  controlBuzzer(1000); 
  digitalWrite(ModeLed, prevMode);
}
 
void loop() {
  // Ticker avtomatik ishlaydi.
  
  button2.check();
  controlMoist();  
}

// **********************************************************************************
//           BUTTON HANDLER FUNKSIYALARI
// **********************************************************************************

void button1Handler(AceButton* button, uint8_t eventType, uint8_t buttonState) {
  Serial.println("EVENT1 - Relay Tugmasi");
  switch (eventType) {
    case AceButton::kEventReleased:
      digitalWrite(RelayPin, !digitalRead(RelayPin));
      toggleRelay = digitalRead(RelayPin);
      break;
  }
}

void button2Handler(AceButton* button, uint8_t eventType, uint8_t buttonState) {
  Serial.println("EVENT2 - Mode Tugmasi");
  switch (eventType) {
    case AceButton::kEventReleased:
      if(prevMode && toggleRelay == HIGH){
        digitalWrite(RelayPin, LOW);
        toggleRelay = LOW;
      }
      prevMode = !prevMode;
      currMode = prevMode ? "A" : "M";
      digitalWrite(ModeLed, prevMode);
      controlBuzzer(500);
      break;
  }
}

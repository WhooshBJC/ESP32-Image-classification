#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_camera.h"
#include "esp_wifi.h"
#include "esp_websocket_client.h"
#include "esp_system.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "img_converters.h"
#include "soc/soc.h" //disable brownout problems
#include "soc/rtc_cntl_reg.h" //disable brownout problems
#include "driver/gpio.h"


// configuration for AI Thinker Camera board
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13


#define wifi_ssid      "gor gor" // CHANGE HERE
#define wifi_password  "#insisttrue" // CHANGE HERE

static const char *TAG = "WS_CLIENT";
esp_websocket_client_handle_t client;

camera_fb_t * fb = NULL;
size_t _jpg_buf_len = 0;
uint8_t * _jpg_buf = NULL;
uint8_t state = 0;

static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data){
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "Connected to WebSocket server");
            break;
        case WEBSOCKET_EVENT_DATA:
            ESP_LOGI(TAG, "RECEIVED: %.s", data->data_len, (char *)data->data_ptr);
            break;
        case WEBSOCKET_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "Disconnected");
            break;
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data){
  if(event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START){
    ESP_LOGI("WIFI", "wifi started, connecting...");
    esp_wifi_connect();
  }

  else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED){
    ESP_LOGW("WIFI", "Disconnected, retrying...");
    esp_wifi_connect();
  }

  else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP){
    ip_event_got_ip_t *event = (ip_event_got_ip_t*)event_data;
    ESP_LOGI("WIFI", "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
  }
}

void onMessageCallback(void *arg, esp_websocket_event_data_t *data){
  printf("Received message: %.*s\n", data->data_len, (char *)data->data_ptr);
}

static void wifi_init(){
  ESP_ERROR_CHECK(nvs_flash_init());
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();
  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));
  ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
  ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));
  wifi_config_t wifi_config = {};
  strcpy((char*)wifi_config.sta.ssid, wifi_ssid);
  strcpy((char*)wifi_config.sta.password, wifi_password);
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());
  ESP_LOGI("WIFI", "Wi-Fi iniialisation complete. Connesting to SSID: %s", wifi_ssid);
}

esp_err_t init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 30000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // parameters for image quality and size
  config.frame_size = FRAMESIZE_VGA; // FRAMESIZE_ + QVGA|CIF|VGA|SVGA|XGA|SXGA|UXGA
  config.jpeg_quality = 15; //10-63 lower number means higher quality
  config.fb_count = 2;
  
  
  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    printf("camera init FAIL: %x\n", err);
    return err;
  }
  sensor_t * s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);
  s->set_quality(s, 30);
  printf("camera init OK");
  return ESP_OK;
};

void camera_task(void *pvParameters) {
    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE("CAMERA", "Image capture failed");
            esp_restart();
        }

        // Send image via WebSocket
        if (esp_websocket_client_is_connected(client)) {
            esp_websocket_client_send_bin(client, (const char *)fb->buf, fb->len, portMAX_DELAY);
            ESP_LOGI("CAMERA", "Image sent, size: %d bytes", fb->len);
        } else {
            ESP_LOGW("CAMERA", "WebSocket not connected");
        }

        esp_camera_fb_return(fb);
        vTaskDelay(pdMS_TO_TICKS(1000)); // Send every 1 second
    }
}


extern "C" void app_main(){
    ESP_LOGI("MAIN","Hello from esp-idf");
    init_camera();
    wifi_init();

    esp_websocket_client_config_t websocket_cfg = {};
    websocket_cfg.uri = "ws://10.244.139.21"; // Your host and port combined
    websocket_cfg.port = 8008;

    client = esp_websocket_client_init(&websocket_cfg);
    esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY, websocket_event_handler, NULL);
    esp_websocket_client_start(client);
    xTaskCreate(camera_task, "camera_task", 4096, NULL, 5, NULL);
}


//void loop() {
  //if (client.available()) {
    //camera_fb_t *fb = esp_camera_fb_get();
    //if (!fb) {
      //Serial.println("img capture failed");
      //esp_camera_fb_return(fb);
      //ESP.restart();
    //}
    //client.sendBinary((const char*) fb->buf, fb->len);
    //Serial.println("image sent");
    //esp_camera_fb_return(fb);
    //client.poll();
  //}
//}
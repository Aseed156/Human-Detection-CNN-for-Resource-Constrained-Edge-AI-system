#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_camera.h"
#include "model_quantized.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"



#define IMG_W      64
#define IMG_H      64
#define IMG_C      3
#define INPUT_SIZE (IMG_W * IMG_H * IMG_C)  // 12288

#define THRESHOLD  0.5f
#define LED_PIN    33


// camera pins
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22


// tflm globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter*     error_reporter = &micro_error_reporter;
const tflite::Model*       model          = nullptr;
tflite::MicroInterpreter*  interpreter    = nullptr;
TfLiteTensor*              input          = nullptr;
TfLiteTensor*              output         = nullptr;

constexpr int kTensorArenaSize = 300 * 1024;
uint8_t* tensor_arena = nullptr;


// camera init 3660
bool initCamera()
{
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;

    // 24MHz camera clk
    config.xclk_freq_hz = 24000000;

   //camera receiving configs
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size   = FRAMESIZE_QQVGA;  
    config.jpeg_quality = 12;             
    config.fb_count     = 2;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }

    // OV2640 post-init tuning
    sensor_t* s = esp_camera_sensor_get();
    if (s != nullptr) {
        s->set_brightness(s, 1);  
        s->set_contrast(s, 1);   
        s->set_saturation(s, 0);   
        s->set_whitebal(s, 1);    
        s->set_exposure_ctrl(s, 1); 
        s->set_gain_ctrl(s, 1);    
    }

    Serial.println("Camera OK");
    return true;
}


// Preprocessing (64 by 64) normalized
void preprocess(camera_fb_t* fb, float* dst)
{
    uint16_t* src = (uint16_t*)fb->buf;
    int srcW = fb->width;   // 160
    int srcH = fb->height;  // 120
    int idx  = 0;

    for (int y = 0; y < IMG_H; y++) {
        int sy = y * srcH / IMG_H;
        for (int x = 0; x < IMG_W; x++) {
            int      sx = x * srcW / IMG_W;
            uint16_t px = src[sy * srcW + sx];

            
            dst[idx++] = ((px >> 11) & 0x1F) * (255.0f / 31.0f);  // R
            dst[idx++] = ((px >>  5) & 0x3F) * (255.0f / 63.0f);  // G
            dst[idx++] = ( px        & 0x1F) * (255.0f / 31.0f);  // B
        }
    }
}


//Setup
void setup()
{
    
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    Serial.begin(115200);
    delay(1000);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    
    Serial.println("  ESP32-CAM Human Detection");
    Serial.println("  by");
    Serial.println("    Aseed Faisal (FA23-BCE-022)");
    Serial.println("    Romisa Malik (FA23-BCE-087)");
    Serial.println("    M. Umer (FA23-BCE-071)");
    

    // Check PSRAM
    if (!psramFound()) {
        Serial.println("ERROR: PSRAM not found!");
        
        while (true) delay(1000);
    }
    Serial.printf("PSRAM free: %d KB\n", ESP.getFreePsram() / 1024);

    // Init camera
    if (!initCamera()) {
        Serial.println("HALTED: Camera init failed");
        Serial.println("Check ribbon cable is fully inserted");
        while (true) delay(1000);
    }

    // Loading model
    model = tflite::GetModel(human_detection_quant_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (true) delay(1000);
    }

    // Allocating tensor arena from PSRAM
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    if (!tensor_arena) {
        Serial.println("PSRAM arena allocation failed!");
        while (true) delay(1000);
    }
    Serial.printf("Arena: %d KB allocated in PSRAM\n", kTensorArenaSize / 1024);

    
    static tflite::AllOpsResolver micro_op_resolver;
    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena,
        kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed!");
        Serial.println("Try increasing kTensorArenaSize");
        while (true) delay(1000);
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    // Tensor Flow Info
    Serial.printf("Input  shape: [%d %d %d %d] type: %d\n",
        input->dims->data[0], input->dims->data[1],
        input->dims->data[2], input->dims->data[3],
        input->type);
    Serial.printf("Output shape: [%d %d] type: %d\n",
        output->dims->data[0], output->dims->data[1],
        output->type);

    Serial.println("Starting Inference\n");
}


// Loop
void loop()
{
    // Capturing frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Capture failed");
        delay(300);
        return;
    }

    // Allocating input buffer from PSRAM
    float* input_buf = (float*)heap_caps_malloc(
        INPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);

    if (!input_buf) {
        Serial.println("Input buffer alloc failed");
        esp_camera_fb_return(fb);
        return;
    }

    // Resize + normalisation of RGB565 frame into float buffer
    preprocess(fb, input_buf);

    // Return camera buffer immediately — frees camera RAM
    esp_camera_fb_return(fb);

    // Copy into model input tensor
    // Handles both float32 and int8 quantized models
    if (input->type == kTfLiteFloat32) {
        memcpy(input->data.f, input_buf, INPUT_SIZE * sizeof(float));
    } else if (input->type == kTfLiteInt8) {
        float scale      = input->params.scale;
        int   zero_point = input->params.zero_point;
        for (int i = 0; i < INPUT_SIZE; i++) {
            input->data.int8[i] = (int8_t)(input_buf[i] / scale + zero_point);
        }
    }

    heap_caps_free(input_buf);

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }

    // Read output 
    float probability = 0.0f;
    if (output->type == kTfLiteFloat32) {
        probability = output->data.f[0];
    } else if (output->type == kTfLiteInt8) {
        probability = (output->data.int8[0] - output->params.zero_point)
                      * output->params.scale;
    }

    // Decision + LED
    if (probability > THRESHOLD) {
        Serial.printf("HUMAN DETECTED: %.2f\n", probability);
        digitalWrite(LED_PIN, HIGH);
    } else {
        Serial.printf("NO HUMAN: %.2f\n", probability);
        digitalWrite(LED_PIN, LOW);
    }

    delay(200);
}

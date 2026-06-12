#include "driver/dac.h"

#define HALL_RA 32
#define HALL_RB 34
#define HALL_RC 35
#define HALL_LA 13
#define HALL_LB 14
#define HALL_LC 27

#define DIR_LEFT GPIO_NUM_2    // Direction control for left motor
#define DIR_RIGHT GPIO_NUM_4   // Direction control for right motor

#define THROTTLE_L DAC_CHANNEL_1
#define THROTTLE_R DAC_CHANNEL_2

#define DAC_MIN    100
#define DAC_MAX    120
#define POLE_PAIRS 15

// ===== OPTIMIZED PID GAINS =====
#define KP 2.5      // Proportional gain
#define KI 0.25     // Integral gain (reduced to prevent spinning)
#define KD 0.8      // Derivative gain

// Left motor PID state
float targetRPM_L = 0;
float currentRPM_L = 0;
float error_L = 0;
float prev_error_L = 0;
float integral_L = 0;
bool movingForward_L = true;

// Right motor PID state
float targetRPM_R = 0;
float currentRPM_R = 0;
float error_R = 0;
float prev_error_R = 0;
float integral_R = 0;
bool movingForward_R = true;

// Current DAC values
int currentDAC_L = 0;
int currentDAC_R = 0;

#define PID_UPDATE_MS 50  // update PID every 50ms
uint32_t lastPID = 0;

volatile long leftTicks   = 0;
volatile long rightTicks  = 0;
volatile uint32_t leftPulses  = 0;
volatile uint32_t rightPulses = 0;

uint32_t lastPrint = 0;

// ===== HALL SENSOR ISR WITH DIRECTION SUPPORT =====
void IRAM_ATTR leftISR()  { 
  leftPulses++;
  if (movingForward_L) {
    leftTicks++;
  } else {
    leftTicks--;  // Negative ticks when reversed
  }
}

void IRAM_ATTR rightISR() { 
  rightPulses++;
  if (movingForward_R) {
    rightTicks++;
  } else {
    rightTicks--;  // Negative ticks when reversed
  }
}

// ===== SET DAC VOLTAGE =====
void setDAC(int l, int r) {
  if (l == 0) dac_output_voltage(THROTTLE_L, 0);
  else        dac_output_voltage(THROTTLE_L, constrain(l, DAC_MIN, DAC_MAX));
  if (r == 0) dac_output_voltage(THROTTLE_R, 0);
  else        dac_output_voltage(THROTTLE_R, constrain(r, DAC_MIN, DAC_MAX));
}

// ===== SET MOTOR DIRECTION =====
void setDirection(bool isLeft, bool forward) {
  if (isLeft) {
    digitalWrite(DIR_LEFT, forward ? 0 : 1);
    movingForward_L = forward;
  } else {
    digitalWrite(DIR_RIGHT, forward ? 0 : 1);
    movingForward_R = forward;
  }
}

// ===== PID CALCULATION =====
int calculatePID(float targetRPM, float currentRPM, 
                 float &error, float &prev_error, float &integral) {
  
  // Calculate error
  error = targetRPM - currentRPM;
  
  // Proportional term
  float P = KP * error;
  
  // Integral term with anti-windup
  integral += error;
  if (integral > 30) integral = 30;
  if (integral < -30) integral = -30;
  
  // Decay integral when close to target
  if (abs(error) < 0.5) {
    integral = integral * 0.9;
  }
  
  float I = KI * integral;
  
  // Derivative term
  float D = KD * (error - prev_error);
  prev_error = error;
  
  // Combine all terms
  float totalAdjustment = P + I + D;
  int dacAdjustment = (int)(totalAdjustment * 0.25);
  
  // Calculate base DAC from target RPM
  int baseDAC = 100 + (int)((targetRPM / 50.0) * 11.0);
  
  // Apply adjustment
  int newDAC = baseDAC + dacAdjustment;
  return constrain(newDAC, 0, DAC_MAX);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

  // ===== HALL SENSOR PINS =====
  pinMode(HALL_LA, INPUT);
  pinMode(HALL_LB, INPUT);
  pinMode(HALL_LC, INPUT);
  pinMode(HALL_RA, INPUT);
  pinMode(HALL_RB, INPUT);
  pinMode(HALL_RC, INPUT);

  // ===== DIRECTION PINS =====
  pinMode(DIR_LEFT, OUTPUT);
  pinMode(DIR_RIGHT, OUTPUT);
  digitalWrite(DIR_LEFT, 0);   // Start forward
  digitalWrite(DIR_RIGHT, 0);  // Start forward

  // ===== HALL SENSOR INTERRUPTS =====
  attachInterrupt(digitalPinToInterrupt(HALL_LA), leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_LB), leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_LC), leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_RA), rightISR, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_RB), rightISR, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_RC), rightISR, RISING);

  // ===== DAC SETUP =====
  dac_output_enable(THROTTLE_L);
  dac_output_enable(THROTTLE_R);
  setDAC(0, 0);

  Serial.println("\n========================================");
  Serial.println("   ARGO MINI PID MOTOR CONTROLLER v3");
  Serial.println("   Forward/Reverse with Negative Odometry");
  Serial.println("========================================");
  Serial.println("Commands:");
  Serial.println("  V <dac_l> <dac_r>  (negative=reverse)");
  Serial.println("  Example: V 5 5      (both forward)");
  Serial.println("  Example: V -5 5     (left reverse, right forward)");
  Serial.println("  S                   (stop)");
  Serial.println("========================================\n");
}

void loop() {
  uint32_t now = millis();

  // ===== SERIAL COMMAND HANDLER =====
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    
    if (line.startsWith("V ")) {
      int spaceIdx = line.indexOf(' ', 2);
      if (spaceIdx > 0) {
        int dacL = line.substring(2, spaceIdx).toInt();
        int dacR = line.substring(spaceIdx + 1).toInt();
        
        // ===== LEFT MOTOR =====
        if (dacL > 0) {
          setDirection(true, true);  // Forward
          targetRPM_L = (dacL) * 5.0;
        } else if (dacL < 0) {
          setDirection(true, false); // Reverse
          targetRPM_L = (abs(dacL)) * 5.0;
        } else {
          targetRPM_L = 0;
        }
        
        // ===== RIGHT MOTOR =====
        if (dacR > 0) {
          setDirection(false, true);  // Forward
          targetRPM_R = (dacR) * 5.0;
        } else if (dacR < 0) {
          setDirection(false, false); // Reverse
          targetRPM_R = (abs(dacR)) * 5.0;
        } else {
          targetRPM_R = 0;
        }
        
        Serial.printf("[CMD] V %d %d -> DAC_L=%d(%s) DAC_R=%d(%s) RPM_L=%.1f RPM_R=%.1f\n",
          dacL, dacR, currentDAC_L, movingForward_L ? "FWD" : "REV",
          currentDAC_R, movingForward_R ? "FWD" : "REV", targetRPM_L, targetRPM_R);
      }
    }
    else if (line == "S") {
      targetRPM_L = 0;
      targetRPM_R = 0;
      currentDAC_L = 0;
      currentDAC_R = 0;
      integral_L = 0;
      integral_R = 0;
      error_L = 0;
      error_R = 0;
      prev_error_L = 0;
      prev_error_R = 0;
      setDAC(0, 0);
      setDirection(true, true);
      setDirection(false, true);
      Serial.println("[CMD] STOP");
    }
  }

  // ===== RESET PID WHEN IDLE =====
  if (targetRPM_L == 0 && targetRPM_R == 0) {
    integral_L = 0;
    integral_R = 0;
    error_L = 0;
    error_R = 0;
    prev_error_L = 0;
    prev_error_R = 0;
    currentDAC_L = 0;
    currentDAC_R = 0;
    setDAC(0, 0);
  }

  // ===== PID UPDATE LOOP =====
  if (now - lastPID >= PID_UPDATE_MS) {
    float elapsed = (now - lastPID) / 1000.0;
    lastPID = now;

    noInterrupts();
    uint32_t lp = leftPulses;  leftPulses = 0;
    uint32_t rp = rightPulses; rightPulses = 0;
    interrupts();

    // Calculate actual RPM from hall sensor pulses
    currentRPM_L = (lp / elapsed) * 60.0 / (POLE_PAIRS * 3);
    currentRPM_R = (rp / elapsed) * 60.0 / (POLE_PAIRS * 3);

    // Apply PID
    currentDAC_L = calculatePID(targetRPM_L, currentRPM_L, 
                                error_L, prev_error_L, integral_L);
    currentDAC_R = calculatePID(targetRPM_R, currentRPM_R, 
                                error_R, prev_error_R, integral_R);

    // Send to motors
    setDAC(currentDAC_L, currentDAC_R);

    // DEBUG OUTPUT
    Serial.printf("PID | Target_L:%.1f Target_R:%.1f | RPM_L:%.1f RPM_R:%.1f | DAC_L:%d DAC_R:%d | Int_L:%.1f Int_R:%.1f\n",
                  targetRPM_L, targetRPM_R, currentRPM_L, currentRPM_R, 
                  currentDAC_L, currentDAC_R, integral_L, integral_R);
  }

  // ===== ODOMETRY & RPM PUBLISH =====
  if (now - lastPrint >= 50) {
    float elapsed = (now - lastPrint) / 1000.0;

    noInterrupts();
    long lt = leftTicks;
    long rt = rightTicks;
    interrupts();

    // Odometry with signed ticks (negative when reversing)
    Serial.printf("O %ld %ld\n", lt, rt);

    // RPM every 500ms
    if (now % 500 < 50) {
      Serial.printf("R %.1f %.1f\n", currentRPM_L, currentRPM_R);
    }

    lastPrint = now;
  }
}

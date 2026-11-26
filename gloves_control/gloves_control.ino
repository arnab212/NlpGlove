// Voice-to-Grasp - Arduino firmware for the two-channel glove controller (paper Section 2.1).
//
// Board : Arduino Mega. Pins 9 and 10 output PWM to the two electronic pressure regulators
//         (channel 1: thumb and index finger; channel 2: other three fingers).
// Input : one text line per command over USB serial (115200 baud): "p1 p2" or "p1,p2",
//         set-points in percent (0-100). Values are clamped to 0-100 % and mapped linearly to
//         a PWM duty value of 0-PWM_MAX (out of 255).
// Reply : "CH1: p1% -> pwm1 | CH2: p2% -> pwm2" for logging.
// Sent by two_regulator_controller.py.

const int PWM1_PIN = 9;   // Regulator 1
const int PWM2_PIN = 10;  // Regulator 2
const int PWM_MIN  = 30;
const int PWM_MAX  = 180;

int pctToPwm(float pct) {
  pct = constrain(pct, 0, 100);
  return (int)round(PWM_MAX  * (pct / 100.0));
}

void setup() {
  pinMode(PWM1_PIN, OUTPUT);
  pinMode(PWM2_PIN, OUTPUT);
  analogWrite(PWM1_PIN, 0);
  analogWrite(PWM2_PIN, 0);

  Serial.begin(115200);
  Serial.setTimeout(50);
  Serial.println("READY For the commands");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    line.replace(',', ' '); // allow comma or space
    int spaceIdx = line.indexOf(' ');
    if (spaceIdx < 0) return;

    float p1 = line.substring(0, spaceIdx).toFloat();
    float p2 = line.substring(spaceIdx + 1).toFloat();

    int pwm1 = pctToPwm(p1);
    int pwm2 = pctToPwm(p2);

    analogWrite(PWM1_PIN, pwm1);
    analogWrite(PWM2_PIN, pwm2);

    Serial.print("CH1: "); Serial.print(p1); Serial.print("% -> "); Serial.print(pwm1);
    Serial.print(" | CH2: "); Serial.print(p2); Serial.print("% -> "); Serial.println(pwm2);
  }
}

#define up_button   8
#define down_button A1
#define left_button 9
#define right_button    12

#define stick_button A0
#define level_stick A3
#define vertical_stick A2

#define deadzoneVmax 510
#define deadzoneVmin 504
#define deadzoneLmax 508
#define deadzoneLmin 502

#define kv1 -0.4971
#define cv1 253.5088
#define kv2 -0.5060
#define cv2 255.0000
#define kl1 -0.4951
#define cl1 251.5340
#define kl2 -0.5080
#define cl2 255.0000
#define debugdebug 0


#define Xbee_num 1 // TX num = 1



int vertical_value = 0;   //current value
int level_value = 0;      //current value
int vertical_valuep = 0;  //previous value
int level_valuep = 0;     //previous value
int vertical_value_actual = 0, level_value_actual = 0, last_vertical_value, last_level_value;
int speed = 20;
int RepeatCounter = 0;
int Mod = 1;
int max_tan_speed = 20;

const double wheel_radius = 7 / 2; //Wheel circumference in cm
const double wheel_circumferenc = 7 * 3.141;
const double WheelBase = 40;
double left_speed, right_speed;

uint16_t key_delay_time = 10;   // for ispressed ()

								//
void setup() {
	pinMode(stick_button, INPUT);
	pinMode(level_stick, INPUT);
	pinMode(vertical_stick, INPUT);

	pinMode(up_button, INPUT);
	pinMode(down_button, INPUT);
	pinMode(left_button, INPUT);
	pinMode(right_button, INPUT);

	Serial1.begin(115200);
	Serial.begin(115200);
	delay(1000);

}

//
void loop() {
	vertical_value = analogRead(vertical_stick);
	level_value = analogRead(level_stick);
	SetSpeed();
	RemapInput();
	PrintOuput();
}

//check button

boolean Read_Debounce(uint8_t key) {
	if (digitalRead(key) == 0) {
		delay(key_delay_time);
		if (digitalRead(key) == 0)
			return true;
	}
	return false;
}

void SetSpeed() {
	if (Read_Debounce(up_button)) max_tan_speed = 10;
	if (Read_Debounce(down_button)) max_tan_speed = 20;
	if (Read_Debounce(left_button)) Mod = 0;
	if (Read_Debounce(right_button)) Mod = 1;
	if (Read_Debounce(stick_button)) max_tan_speed = 50;
}

void RemapInput() {
	if (vertical_value >= deadzoneVmax) {
		vertical_value_actual = kv1*vertical_value + cv1;
	}
	else if (vertical_value <= deadzoneVmin) {
		vertical_value_actual = kv2*vertical_value + cv2;
	}
	else {
		vertical_value_actual = 0;
	}

	if (level_value >= deadzoneLmax) {
		level_value_actual = kl1*level_value + cl1;
	}
	else if (level_value <= deadzoneLmin) {
		level_value_actual = kl2*level_value + cl2;
	}
	else {
		level_value_actual = 0;
	}
}





void PrintOuput() {
	if (last_vertical_value != vertical_value_actual || last_level_value != level_value_actual) RepeatCounter = 0;
	if (RepeatCounter != 2) {
		if (last_vertical_value == vertical_value_actual && last_level_value == level_value_actual) RepeatCounter++;
		char buffer_out[40];
		TurnMix(); char buffer1[10],buffer2[10];
		

		if (-1< left_speed && left_speed< 1 && -1 < right_speed && right_speed< 1) { left_speed = 0;right_speed = 0; } //Limite center dead zone within +-1


		snprintf(buffer_out, 20, "S%dM%dL%sR%sD0E", Xbee_num, Mod, dtostrf(left_speed,5,2, buffer1), dtostrf(right_speed,5,2, buffer2)); //duration =0sec, D0
		//snprintf(buffer_out, 20, "S%dM%dL%dR%dD0E", Xbee_num, Mod, left_speed,right_speed); //duration =0sec, D0
		//snprintf(buffer_out, 20, "L%dR%d", left_speed, right_speed);

#if debugdebug==0
		delay(100);
		Serial1.println(buffer_out);Serial.println(buffer_out);	
#endif
	}
		last_vertical_value = vertical_value_actual;last_level_value = level_value_actual;
}

void TurnMix() {
	double Speed_forward, Speed_rad;
	double controller_angle;
	if (vertical_value_actual != 0 && Mod)
	{

		controller_angle = atan2(vertical_value_actual, level_value_actual);

		if (controller_angle >= 0)
		{
			Speed_rad = (controller_angle - 3.1415926 / 2) / (3.1415926 / 2)*(max_tan_speed * 2 / WheelBase);
		}
		else {
			Speed_rad = (controller_angle + 3.1415926 / 2) / (3.1415926 / 2)*(max_tan_speed * 2 / WheelBase);
		}
	}
	else {
		Speed_rad = 0;
	}
	Speed_forward = vertical_value_actual / 255.0*max_tan_speed;

	left_speed = (2 * Speed_forward - WheelBase*Speed_rad) / 2; right_speed = (2 * Speed_forward + WheelBase*Speed_rad) / 2;

	if (vertical_value_actual == 0) {
		if (level_value_actual > 0) {
			left_speed = level_value_actual / 255.0*max_tan_speed; right_speed = -level_value_actual / 255.0*max_tan_speed;
		}
		else if (level_value_actual < 0) {
			left_speed = level_value_actual / 255.0*max_tan_speed; right_speed = -level_value_actual / 255.0*max_tan_speed;
		}
	}

	if (( -5*3.141/180<controller_angle&&controller_angle<0)|| (-180*3.141/180<controller_angle&&controller_angle<-175*3.141/180)) {
		left_speed = 0;right_speed = 0;// negative angle deadzone
	}



#if debugdebug
	Serial.print("Controller angle ");Serial.println(controller_angle*180/3.14);
	Serial.print("Left Right speed ");Serial.print(left_speed);Serial.print(" "); Serial.println(right_speed);
#endif
}
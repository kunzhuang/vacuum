#include <PID_v1.h>
#include "math.h"


// Motor shield pin definition
#define  E1  5
#define  M1 4
#define  E2  6
#define  M2 7
#define  WheelBase 40.0 // Wheel base length in cm

#define debugdebug 0

// Wheel parameters 
volatile int PulsecountL, PulsecountR;
long timestamp_last, timestamp_now;
int rpm = 0; 
const double wheel_radius = 7 / 2; //Wheel circumference in cm
const double wheel_circumferenc = 7 * 3.141;
const double gear_ratio = 63.6025 * 4; //Encoder ratio to wheel
const int sampling_time_const = 5;//50;
double wheel_speed_L, wheel_speed_R;
int time_interival;
String strCommand, strCommand1;
int Serial_port=0;


int counter = 0;

unsigned int Xbee_num, Mod;
float Duration_clock=0, Duration = 0;
double Left_speed, Right_speed;
boolean Left_dir=0, Right_dir=0; // 0 forwared 1 backward



double P = 0.5, I =20, D = 0.000;
double Setpointleft = 0, Setpointright = 0;
double Inputleft, Outputleft, Inputright, Outputright;
double Setpointleft_break_last, Setpointright_break_last, Setpointleft_TurnMix_last, Setpointright_TurnMix_last;

PID pidLeft(&Inputleft, &Outputleft, &Left_speed, P, I, D, REVERSE);PID pidRight(&Inputright, &Outputright, &Right_speed, P, I, D, REVERSE);

boolean ApplybreakL, ApplybreakR;



void setup()
{
	Serial.begin(115200);
	Serial1.begin(115200);
	Serial2.begin(115200);
	pinMode(M1, OUTPUT);pinMode(M2, OUTPUT);
	attachInterrupt(3, RPMcounterL, FALLING);	attachInterrupt(2, RPMcounterR, FALLING);
	pidLeft.SetOutputLimits(-255, 255); pidRight.SetOutputLimits(-255, 255);
	pidLeft.SetSampleTime(sampling_time_const); pidRight.SetSampleTime(sampling_time_const);
	pidLeft.SetMode(AUTOMATIC);pidRight.SetMode(AUTOMATIC);
}

void loop()
{
	Receiver();
	wheel_speed();
	Inputleft = wheel_speed_L;Inputright = wheel_speed_R;
	pidLeft.Compute();pidRight.Compute();
	Motor_out();


	//Serial.print("OL");Serial.print(Outputleft);Serial.print("OR");Serial.println(Outputright);

#if debugdebug
	
#endif

	//BreakLeft();BreakRight();
}

void Motor_out() {

	if (Outputleft >= 0) {
		Left_dir = 1;
		digitalWrite(M1, Left_dir);analogWrite(E1, Outputleft);
	}else{
		Left_dir = 0;
		digitalWrite(M1, Left_dir);analogWrite(E1, abs(Outputleft));
	}
	
	if (Outputright >= 0) {
		Right_dir = 1;
		digitalWrite(M2, Right_dir);analogWrite(E2, Outputright);
	}
	else {
		Right_dir = 0;
		digitalWrite(M2, Right_dir);analogWrite(E2, abs(Outputright));
	}


}

void Receiver()
{	/*
	Receiver function
	It takes input format S0M1L10R10D50000E, S: Serial port or statr, M mode reserved input, L left set speed in cm/s, R right set speed in cm/s, D duration in mili sections max long, E end
		Serial wired serial connection (USB), monitoring any command that duration >0
		Serial1 Xbee 1, should be S1 or to joystick controller
		Serial2 Xbee 2, should be S2, monitoring any command that duration >0
		Serial1 can interrupt inputs from Serial or Serial2
			Any commands which contain duration >0 will be executed sequentially

	*/
	if (Duration <= 0) {
		while (Serial2.available() > 0) {
			char someChar = Serial2.read(); //read character one by one to somechar, only read once in one loop
			//delay(1);
			strCommand.concat(someChar); // append somechar to strCommand
			if (someChar == 'E') {//str.concat("End found");
				//Serial.end();
				unsigned S = strCommand.indexOf('S'); // get location of S
				unsigned M = strCommand.indexOf('M'); // get location of M
				unsigned L = strCommand.indexOf('L'); //......
				unsigned R = strCommand.indexOf('R');
				unsigned D = strCommand.indexOf('D');
				unsigned E = strCommand.indexOf('E');
				Serial_port = strCommand.substring(S + 1, M).toFloat();// Extract port number
				Left_speed = strCommand.substring(L + 1, R).toFloat();Right_speed = strCommand.substring(R + 1, D).toFloat();// Extract left speed and right speed
				Duration = strCommand.substring(D + 1, E).toFloat();// Extract duration
				strCommand = ""; // reset strcommand
				//Serial.begin(115200);
				Duration_clock = millis() + Duration; // current time + duration
				Serial2.print(Duration);
				break;
			}
		}
	}
	else { // times up, set left speed and right speed to 0
		Duration = Duration_clock - millis(); //calculate time left
		Serial2.print(Duration);
		if (Duration <= 0) { Left_speed = 0;Right_speed = 0;
		}
	}

	while (Serial1.available() > 0) {// Read inputs from remote controller
		char someChar = Serial1.read();
		//delay(1);
		strCommand1.concat(someChar);
		if (someChar == '\n') {//str.concat("End found");
			//Serial1.end();
			unsigned S = strCommand1.indexOf('S');
			unsigned M = strCommand1.indexOf('M');
			unsigned L = strCommand1.indexOf('L');
			unsigned R = strCommand1.indexOf('R');
			unsigned D = strCommand1.indexOf('D');
			unsigned E = strCommand1.indexOf('E');
			Serial_port = strCommand1.substring(S + 1, M).toFloat();
			Left_speed = strCommand1.substring(L + 1, R).toFloat();	Right_speed = strCommand1.substring(R + 1, D).toFloat();
			Duration = strCommand1.substring(D + 1, E).toFloat();
			strCommand1 = "";
			//Serial1.begin(115200);
			break;
		}
}	
		//Serial.print(Serial_port);Serial.print("  ");Serial.print(Left_speed);Serial.print("  ");Serial.print(Right_speed);Serial.print("  ");Serial.println(Duration);
}



void wheel_speed()
{
	/*
	This fucntion is used to calculate wheel speed
	*/
	delayMicroseconds(sampling_time_const*1000); // Sampling period
	detachInterrupt(3);detachInterrupt(2); //L R Reject any interrupts during the calculation
	timestamp_now = micros();
	time_interival = (timestamp_now - timestamp_last)/1000;
	wheel_speed_L = PulsecountL / ((time_interival) / 1000.0) / gear_ratio*wheel_circumferenc; //Left wheel speed in cm/s
	wheel_speed_R = PulsecountR / ((time_interival) / 1000.0) / gear_ratio*wheel_circumferenc; //Right wheel speed in cm/s
	
	if (Left_dir) wheel_speed_L = -wheel_speed_L;
	if (Right_dir) wheel_speed_R = -wheel_speed_R;

	PulsecountL = 0; PulsecountR = 0;
	timestamp_last = micros();
	attachInterrupt(3, RPMcounterL, FALLING);attachInterrupt(2, RPMcounterR, FALLING); // Store interrupt status
	Serial2.print(" ");
	Serial2.print(wheel_speed_L);Serial2.print(" ");Serial2.println(wheel_speed_R);
}

void RPMcounterL() { PulsecountL++; } // for interrupt calculation left wheel
void RPMcounterR() { PulsecountR++; } // for interrupt calculation right wheel


void BreakLeft()
{		//ApplybreakL=true;
	if (Setpointleft == 0 && Setpointleft_break_last != 0)
	{
		if (Left_dir)
		{
			digitalWrite(M1, LOW);analogWrite(E1, 100);delay(10);analogWrite(E1, 0);//ApplybreakL=false;
		}
		else
		{
			digitalWrite(M1, HIGH);analogWrite(E1, 100);delay(10);analogWrite(E1, 0);//ApplybreakL=false;
		}
		//analogWrite(E1, 0);
	}
	Setpointleft_break_last = Setpointleft;
}

void BreakRight()
{
	if (Setpointright == 0 && Setpointright_break_last != 0)
	{
		if (Right_dir)
		{
			digitalWrite(M2, LOW);analogWrite(E2, 100);delay(10);analogWrite(E2, 0);//ApplybreakL=false;
		}
		else
		{
			digitalWrite(M2, HIGH);analogWrite(E2, 100);delay(10);analogWrite(E2, 0);//ApplybreakL=false;
		}
		//analogWrite(E1, 0);
	}
	Setpointright_break_last = Setpointright;
}


void Receiver1() // Not used in this program, will not be called
{
	while (Serial1.available() > 0) {
		char someChar = Serial1.read();
		//delay(1);
		strCommand.concat(someChar);
		if (someChar == 'E') {//str.concat("End found");
			unsigned S = strCommand.indexOf('S');
			unsigned M = strCommand.indexOf('M');
			unsigned L = strCommand.indexOf('L');
			unsigned R = strCommand.indexOf('R');
			unsigned D = strCommand.indexOf('D');
			unsigned E = strCommand.indexOf('E');
			Left_speed = strCommand.substring(L + 1, R).toFloat();
			Right_speed = strCommand.substring(R + 1, D).toFloat();
			Duration = strCommand.substring(D + 1, E).toFloat();
			strCommand = "";


			//Serial.print(Left_speed);Serial.print("  ");Serial.println(Right_speed);

#if debugdebug
			strCommand.concat("End found");
			Serial.println(strCommand);
			Serial.print(Left_speed);Serial.print("  ");Serial.println(Right_speed);

#endif
			break;
		}

		if (Left_speed > 0) { Left_dir = 1; }
		else { Left_dir = 0; }


		if (Right_speed > 0) { Right_dir = 1; }
		else { Right_dir = 0; }

	}
}

#include <Servo.h>
#include <ros.h>
#include <std_msgs/Int16MultiArray.h>
#define triggerPin 11

Servo servo1;
Servo servo2;


ros::NodeHandle nh;
int prev_servo = 90;

void servoCallback(const std_msgs::Int16MultiArray& msg) {
  if (msg.data_length >= 2) {
    servo1.write(msg.data[0]);
    if (msg.data[1] < prev_servo){
      for(int i = prev_servo; i >= msg.data[1]; i--){
        servo2.write(i);
        delay(30);
      }
      prev_servo = msg.data[1];
    }
    else{
      servo2.write(msg.data[1]);
    }
    if(msg.data[2] == 1){
      digitalWrite(triggerPin,LOW);
      delay(1000);
      digitalWrite(triggerPin,HIGH);
    }
    else if(msg.data[2] == 0){
      digitalWrite(triggerPin,HIGH);
    }
  }
}

ros::Subscriber<std_msgs::Int16MultiArray> sub("servo_angles", servoCallback);

void setup() {
  servo1.attach(9);  // Connect servo1 to pin 9
  servo2.attach(10); // Connect servo2 to pin 10
  pinMode(triggerPin,OUTPUT);
  digitalWrite(triggerPin,HIGH);

  nh.initNode();
  nh.subscribe(sub);
}

void loop() {
  nh.spinOnce();
  delay(100);
}

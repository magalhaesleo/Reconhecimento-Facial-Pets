#include "Servo.h"
#include <stdio.h>
#include <wiringPi.h>
#include <softPwm.h>
#include <chrono>

#define PIN 1

using namespace std;

bool initialized = false;
auto lastRotation = chrono::_V2::system_clock::now();

Servo::Servo()
{
}

Servo::~Servo()
{
    //dtor
}

void Servo::RotateDispenser()
{
    if (initialized)
    {
        auto now = chrono::_V2::system_clock::now();
        auto c = chrono::duration_cast<chrono::hours>(now - lastRotation).count();

        printf("seconds elapsed: %lli\n", c);
        if (c < 4)
        {
            return;
        }

        lastRotation = now;
    }
    else
    {
        initialized = true;
    }

        printf ("Raspberry Pi wiringPi PWM test program\n") ;

        if (wiringPiSetup () == -1)
        {
            printf("Falha ao iniciar wiringPi");
        	return;
        }

        pinMode (PIN, PWM_OUTPUT) ;
        digitalWrite(PIN, 0);

        softPwmCreate(PIN, 0, 1024);

        printf("13\n");
        softPwmWrite(PIN, 6);
        delay(600);
        softPwmWrite(PIN, 15);
        delay(400);
        printf("17\n");
        softPwmWrite(PIN, 24);
        delay(440);
        softPwmStop(PIN);
}

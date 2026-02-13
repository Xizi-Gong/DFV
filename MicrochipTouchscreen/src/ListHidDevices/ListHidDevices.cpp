/** *********************************************
* \file
* \brief		Example to use MaxTouchDevice
*
* \author		Yizhong Zhang
* \date			1/11/2022
*********************************************  */
#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include <MaxTouch.h>
#include <yzLib/yz_lib.h>

HIDBase hid_base;


int main() {
	hid_base.Init();

	hid_base.PrintAllHIDDevices();

	hid_base.Exit();

	return 0;
}

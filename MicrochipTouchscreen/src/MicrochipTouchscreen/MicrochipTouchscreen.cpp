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
#include <cstring>   // std::memcpy
#include <memory>    // std::unique_ptr
#include <MaxTouch.h>
#include <yzLib/yz_lib.h>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <zmq.hpp>  // from cppzmq

yz::opengl::DemoWindowManager	manager;
yz::opengl::DemoWindow2D		win2d;

MaxTouchDevice					mxt_device;
float							query_fps;

int								prev_raw_frame_id = 0;
int								raw_frame_id = 0;
unsigned long long				raw_timestamp_sys = 0;
int								raw_x = 0, raw_y = 0;
std::vector<short>				raw_data;

std::vector<MaxTouchDevice::TouchPoint>		touch_points;

void print2d() {
	glColor3f(1, 0, 0);
	yz::opengl::printInfo(0, 0, "frame_id:%d, x:%d, y:%d, fps:%f", raw_frame_id, raw_x, raw_y, query_fps);
	yz::opengl::printInfo(0, 20, "timestamp:%lld", raw_timestamp_sys);

	int line_y = 60;
	yz::opengl::printInfo(
		0, line_y += 20, "TouchPoints");

	//	print all information of each points
	for (int i = 0; i < touch_points.size(); i++) {
		int touch_type, touch_event;
		bool touched = MaxTouchDevice::TouchStatus(
			touch_type, touch_event, touch_points[i].touch_status);

		if (touched)
			yz::opengl::setSequentialDisplayColor(i);
		else
			glColor3f(0, 0, 0);

		yz::opengl::printInfo(
			10, line_y += 20,
			"%d %d: %.2f, %.2f, %d, %d, %d, %d, %d, %d",
			touch_type,
			touch_event,
			touch_points[i].x_normalize,
			touch_points[i].y_normalize,
			int(touch_points[i].vect1),
			int(touch_points[i].vect0),
			int(touch_points[i].amplitude),
			int(touch_points[i].area),
			int(touch_points[i].width_mm),
			int(touch_points[i].height_mm)
		);
	}

}

void draw2d() {
	yz::opengl::drawXYZAxis();

	glColor3f(0, 0, 0);
	yz::opengl::drawAABBWire(
		yz::Vec2f(0, 0),
		yz::Vec2f(mxt_device.size_x_mm, mxt_device.size_y_mm) * 0.001f
	);

	//	draw raw data
	if (raw_x && raw_y) {
		float dx = mxt_device.size_x_mm / 1000. / raw_x;
		float dy = mxt_device.size_y_mm / 1000. / raw_y;

		glBegin(GL_QUADS);
		for (int i = 0; i < raw_x; i++) {
			for (int j = 0; j < raw_y; j++) {
				unsigned char rgb[3];
				yz::utils::convertToColorGray(rgb, raw_data[i * raw_y + j], -1000, 1000);

				glColor3ubv(rgb);
				yz::Vec2f v[4] = {
					yz::Vec2f(i * dx, j * dy),
					yz::Vec2f((i + 1) * dx, j * dy),
					yz::Vec2f((i + 1) * dx, (j + 1) * dy),
					yz::Vec2f(i * dx, (j + 1) * dy),
				};
				for (int k = 0; k < 4; k++)
					glVertex2f(v[k].x, v[k].y);
			}
		}
		glEnd();
	}

	//	draw touch points
	//if (touch_points.size()) {
	//	float x_size = mxt_device.size_x_mm / 1000.;
	//	float y_size = mxt_device.size_y_mm / 1000.;

	//	glLineWidth(3);
	//	for (int i = 0; i < touch_points.size(); i++) {
	//		if (touch_points[i].touch_status & 0x80) {
	//			//	draw touch point as circle
	//			float x = x_size * touch_points[i].x_normalize;
	//			float y = y_size * touch_points[i].y_normalize;
	//			float radius = 0.01 * touch_points[i].amplitude / 128;
	//			yz::Vec2f p(x, y);

	//			yz::opengl::setSequentialDisplayColor(i);
	//			yz::opengl::drawPointAsCircle(p, radius);

	//			//	draw touch direction
	//			float angle_deg, magnitude;
	//			MaxTouchDevice::TouchAngleMagnitude(
	//				angle_deg, magnitude,
	//				touch_points[i].vect1, touch_points[i].vect0);
	//			yz::Vec2f r(0, magnitude * 0.002);
	//			r.SetRotateDeg(angle_deg);
	//			yz::opengl::drawLineSegment(p + r, p - r);
	//		}
	//	}
	//	glLineWidth(1);
	//}

}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		mxt_device.Close();
		exit(0);
	}
}

void idle() {
	mxt_device.GetRawData(raw_data, raw_x, raw_y, raw_frame_id, raw_timestamp_sys);
	mxt_device.GetTouchData(touch_points);

	if (raw_frame_id != prev_raw_frame_id) {
		static yz::windows::RealTimeFPSCalculator fps_calc;
		query_fps = fps_calc.GetFPS();
		prev_raw_frame_id = raw_frame_id;
	}

}

int main() {
	int touch_points = 0;

	 if (!mxt_device.Open(8, touch_points)) {
	 	return 0;
	 }

	 //win2d.keyboardFunc = keyboard;
	 //win2d.SetDraw(draw2d);
	 //// win2d.SetDrawAppend(print2d);
	 //win2d.CreateGLUTWindow();

	 //manager.AddIdleFunc(idle);
	 //manager.AddIdleFunc(win2d.idleFunc);
	 //manager.EnterMainLoop();

	 //return 0;
	
    //if (!mxt_device.Open(8, touch_points)) {
    //    return 0;
    //}

    // ZeroMQ publisher setup
    zmq::context_t ctx{1};
    zmq::socket_t pub{ctx, zmq::socket_type::pub};
    pub.set(zmq::sockopt::sndhwm, 10);
    // pub.set(zmq::sockopt::conflate, 1); // uncomment if you only need the latest frame
    pub.bind("tcp://*:5556");
    std::cout << "ZMQ PUB bound to tcp://*:5556 (topic='frame')\n";

    prev_raw_frame_id = -1;

    while (true) {
        // Read latest data from the device
        mxt_device.GetRawData(raw_data, raw_x, raw_y, raw_frame_id, raw_timestamp_sys);

        // Only publish on a new frame
        if (raw_x > 0 && raw_y > 0 && !raw_data.empty() && raw_frame_id != prev_raw_frame_id) {
            prev_raw_frame_id = raw_frame_id;

            // Tiny JSON header: { seq, timestamp, shape:[H,W], dtype, endianness }
            std::ostringstream oss;
            oss << "{"
                << "\"seq\":" << raw_frame_id << ","
                << "\"timestamp\":" << raw_timestamp_sys << ","
                << "\"shape\":[" << raw_x << "," << raw_y << "],"
                << "\"dtype\":\"int16\","
                << "\"endianness\":\"little\""
                << "}";

            const std::string topic = "frame";
            const std::string header = oss.str();

            zmq::message_t topic_msg(topic.data(), topic.size());
            zmq::message_t header_msg(header.data(), header.size());
            zmq::message_t raw_msg(raw_data.size() * sizeof(short));
            std::memcpy(raw_msg.data(), raw_data.data(), raw_msg.size());

            // multipart: [topic][header][raw bytes]
            pub.send(topic_msg, zmq::send_flags::sndmore);
            pub.send(header_msg, zmq::send_flags::sndmore);
            pub.send(raw_msg, zmq::send_flags::none);

            // Optional: light logging
            if ((raw_frame_id % 20) == 0) {
                std::cout << "Sent frame seq=" << raw_frame_id
                          << " shape=" << raw_x << "x" << raw_y
                          << " bytes=" << raw_msg.size() << "\n";
            }
        }

        // Small sleep to avoid busy-spin if no new data
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Not reached normally; use Ctrl+C to exit
    mxt_device.Close();
    return 0;
}

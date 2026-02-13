/** *********************************************
* \file
* \brief		MaxTouch Interface, developed on Atmel ATMXT2952Tx
* 
* \author		Yizhong Zhang
* \date			1/11/2022
*********************************************  */
#ifndef MAXTOUCH_H
#define MAXTOUCH_H

#include <Windows.h>
#include <thread>
#include <mutex>
#include <cmath>
#include <hidapi.h>
#pragma comment(lib, "hidapi.lib")

#include "info_block.h"


/**
* Interface to access HID
* 
* Since functions of hid is not complex, so this class is a simple wrap only
*/
class HIDBase {
public:
	int Init() {
		if (hid_init_flag)
			return 1;

		if (hid_init()) {
			std::cout << "Error: HIDBase::Init(), hid_init() failed" << std::endl;
			return 0;
		}

		//	check version
		if (hid_version()->major != HID_API_VERSION_MAJOR ||
			hid_version()->minor != HID_API_VERSION_MINOR ||
			hid_version()->patch != HID_API_VERSION_PATCH)
		{
			std::cout << "Warning: HIDBase::Init(), Compile-time version is different than runtime version of hidapi" << std::endl;
		}

		return 1;
	}

	int Exit() {
		if (!hid_init_flag)
			return 1;

		if (handle)
			Close();

		if (hid_exit()) {
			std::cout << "Error: HIDBase::Exit(), hid_exit() failed" << std::endl;
			return 0;
		}

		return 1;
	}

	int Open(
		unsigned short vendor_id,
		unsigned short product_id,
		const wchar_t* serial_number = NULL,
		const wchar_t* product_string = NULL
	) {
		if (handle) {
			std::cout << "error: HIDBase::Open(), already opened" << std::endl;
			return 0;
		}

		if (!Init())
			return 0;

		//	if serial number exist, open the HID device directly
		if (serial_number) {
			handle = hid_open(vendor_id, product_id, serial_number);
			if (!handle) {
				std::cout << "error: HIDBase::Open(), failed to open" << std::hex << vendor_id << ", " << product_id << std::endl;
				return 0;
			}
		}
		//	open HID device withouth serial number, iterate (vendor_id, product_id) to get the device
		else {
			//	count number of devices with the same (vendor_id, product_id)
			int device_count = 0;
			struct hid_device_info* devs = hid_enumerate(vendor_id, product_id);
			struct hid_device_info* cur_dev = devs;
			while (cur_dev) {
				device_count++;
				cur_dev = cur_dev->next;
			}

			//	check whether target device exist
			if (!device_count) {
				std::cout << "error: HIDBase::Open(), no HID device " << std::hex << vendor_id << ", " << product_id << std::endl;
				hid_free_enumeration(devs);
				return 0;
			}

			//	iterate all devices to get the target
			cur_dev = devs;
			while (cur_dev) {
				//	product_string provided, try to find the device
				if (product_string && wcscmp(product_string, cur_dev->product_string) == 0) {					
					handle = hid_open_path(cur_dev->path);
					if (!handle) {
						std::cout << "error: HIDBase::Open(), failed to open " << cur_dev->path << std::endl;
						hid_free_enumeration(devs);
						return 0;
					}
				}
				//	product_string not provided, try the first device
				else if (!product_string && !handle) {
					handle = hid_open_path(cur_dev->path);
				}
				cur_dev = cur_dev->next;
			}

			//	cannot open device with given product_string
			if (!handle && product_string) {
				std::cout << "error: HIDBase::Open(), failed to open" << std::hex 
					<< vendor_id << ", " << product_id << ", " << product_string << std::endl;
				std::cout << "\texisting HID devices:" << std::endl;
				cur_dev = devs;
				while (cur_dev) {
					printf("\t\t%ls\n", cur_dev->product_string);
				}
				cur_dev = cur_dev->next;
			}

			hid_free_enumeration(devs);
		}

		if (!handle) {
			std::cout << "error: HIDBase::Open(), failed to open" << std::hex << vendor_id << ", " << product_id << std::endl;
			return 0;
		}

		ReadHIDInfo();

		std::cout << "Opened HID interface: " << std::hex << vendor_id << ", " << product_id << std::endl;
		PrintHIDInfo();

		return 1;
	}

	int Close() {
		if (!handle) {
			std::cout << "error: HIDBase::Close(), not opened" << std::endl;
			return 0;
		}

		hid_close(handle);
		handle = NULL;

		return 1;
	}

	void PrintAllHIDDevices() {
		bool old_hid_init_flag = hid_init_flag;

		if (!Init())
			return;

		if (true) {
			struct hid_device_info* devs = hid_enumerate(0, 0);
			struct hid_device_info* cur_dev = devs;
			while (cur_dev) {
				printf("Device Found\n  type: %04hx %04hx\n  path: %s\n  serial_number: %ls", cur_dev->vendor_id, cur_dev->product_id, cur_dev->path, cur_dev->serial_number);
				printf("\n");
				printf("  Manufacturer: %ls\n", cur_dev->manufacturer_string);
				printf("  Product:      %ls\n", cur_dev->product_string);
				printf("  Release:      %hx\n", cur_dev->release_number);
				printf("  Interface:    %d\n", cur_dev->interface_number);
				printf("  Usage (page): 0x%hx (0x%hx)\n", cur_dev->usage, cur_dev->usage_page);
				printf("\n");
				cur_dev = cur_dev->next;
			}
			hid_free_enumeration(devs);
		}

		if (!old_hid_init_flag)
			Exit();
	}

public:
	bool			hid_init_flag = false;

	hid_device*		handle = NULL;

	std::wstring	manufacturer;
	std::wstring	product;
	std::wstring	serial_number;
	std::wstring	indexed;

protected:
	void ReadHIDInfo() {
		int res;
		const int MAX_STR = 255;
		wchar_t wstr[MAX_STR + 1];

		// Read the Manufacturer String
		wstr[0] = 0x0000;
		if (hid_get_manufacturer_string(handle, wstr, MAX_STR) < 0)
			printf("Unable to read manufacturer string\n");
		else
			manufacturer = wstr;

		// Read the Product String
		wstr[0] = 0x0000;
		if (hid_get_product_string(handle, wstr, MAX_STR) < 0)
			printf("Unable to read product string\n");
		else
			product = wstr;

		// Read the Serial Number String
		wstr[0] = 0x0000;
		if (hid_get_serial_number_string(handle, wstr, MAX_STR) < 0)
			printf("Unable to read serial number string\n");
		else
			serial_number = wstr;

		// Read Indexed String 1
		wstr[0] = 0x0000;
		if (hid_get_indexed_string(handle, 1, wstr, MAX_STR) < 0)
			printf("Unable to read indexed string 1\n");
		else
			indexed = wstr;
	}

	void PrintHIDInfo() {
		printf("manufacturer: %ls\n", manufacturer.c_str());
		printf("product: %ls\n", product.c_str());
		printf("serial_number: %ls\n", serial_number.c_str());
		printf("indexed: %ls\n", indexed.c_str());
	}
};


/**
* Interface to manipulate MaxTouch device
* 
* guidelines of use:	\n
*	1, call Open() to open the device with specified parameters.	\n
*	2, call GetRawData() and GetTouchData() to get the current raw array and touch points.	\n
*	3, call Close() to terminate read data.	\n
* 
* Two static util functions are given: TouchStatus() and TouchAngleMagnitude(),
* to calculate parameters of touch points. For detailed usage, refer to comments 
* of each function.
*/
class MaxTouchDevice : protected HIDBase {
public:
	struct TouchPoint {
		unsigned long long	timestamp_sys;	///< time stamp of this touch point

		unsigned char	touch_status;	///< status of the touch point, see TouchStatus() for details
		float			x_normalize, y_normalize;	///< normalized position (0-1) of this touch point
		char			vect1, vect0;	///< direction of this touch point, Atmel 40002131A.pdf, pp.58
		unsigned char	amplitude;		///< proportional to touch size and pressure
		unsigned char	area;			///< area of the touch in covered nodes
		unsigned char	width_mm;		///< width of the touch point in millimeter
		unsigned char	height_mm;		///< height of the touch point in millimeter
	};

public:
	/**
	* Start the MaxTouch device. 
	* 
	* \param raw_bits			depth of raw data, 8/16 bits. set to 0 if do not need raw data.
	* \param touch_point_num	max touch points to read, 0-15. set to 0 if do not need touch points.
	* \return					0: error occour.  1: succeed
	*/
	int Open(unsigned int raw_bits = 8, unsigned int touch_point_num = 10, bool diag_flip = false) {
		//	========== Check parameters ==========
		if (touch_point_num > 0x0F) {
			std::cout << "Error: MaxTouchDevice::Open(), unsupported touch number: " << touch_point_num << "max number 15, set to 10" << std::endl;
			touch_point_num = 10;
		}
		touch_timestamp_sys.resize(touch_point_num);
		touch_data.resize(touch_point_num);

		if (raw_bits != 0 && raw_bits != 8 && raw_bits != 16) {
			std::cout << "Error: MaxTouchDevice::Open(), unsupported raw_bits: " << raw_bits << ", 0/8/16 allowed, set to 8 bits" << std::endl;
			raw_bits = 8;
		}
		this->raw_bits = raw_bits;

		if (!touch_point_num && !raw_bits)
			return 0;

		//	========== Initialize HID ==========
		if (!HIDBase::Init())
			return 0;

		if (!HIDBase::Open(0x03EB, 0x214E, NULL, L"Atmel maXTouch Control"))
			return 0;

		hid_set_nonblocking(handle, 0);

		//	========== Initialize MaxTouch ==========
		int clear_pkt_num = ClearHIDBuffer();
		//if (clear_pkt_num) {
		//	std::cout << "Warning: MaxTouchDevice::Open(), cleared existing packets: " << clear_pkt_num << std::endl;
		//}

		ReadInfoBlock();
		PrintInfoBlock();		

		InitAutoReturn(diag_flip);

		//	========== setup flip ==========
		if (diag_flip) {
			diag_flip_flag = true;

			int tmp = size_x_mm;
			size_x_mm = size_y_mm;
			size_y_mm = tmp;
		}

		//	========== Start Query ==========
		if (!touch_data.empty())
			event_touch_data = CreateEventA(NULL, FALSE, FALSE, "MaxTouchDevice_event_touch_data");
		if (this->raw_bits)
			event_raw_data = CreateEventA(NULL, FALSE, FALSE, "MaxTouchDevice_event_raw_data");

		query_flag = true;
		terminate_flag = false;
		std::thread(&MaxTouchDevice::QueryThread, this).detach();

		return 1;
	}

	/**
	* Close the MaxTouch device. 
	*/
	int Close() {
		hid_packet pkt;

		if (!touch_data.empty()) {
			pkt.report_id = 0x01;
			pkt.dbg_cmd = 0x00;
			WritePacket(pkt);
		}

		if (raw_bits) {
			pkt.report_id = 0x01;
			pkt.dbg_cmd = 0xE2;
			WritePacket(pkt);
		}

		query_flag = false;
		while (!terminate_flag)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));

		//	clear data
		touch_data_mutex.lock();
		touch_data.clear();
		touch_data_mutex.unlock();

		raw_data_mutex.lock();
		raw_data.clear();
		raw_data_mutex.unlock();

		HIDBase::Close();

		return 1;
	}

	/**
	* Get raw mutual capacitive touch data.
	* 
	* when a new frame arrives, event_raw_data will be set.
	* 
	* \param raw_data		output the raw 16 bits data. 8 bits data is multplied by 8 before output.
	* \param raw_x			output the dimension x of raw data.
	* \param raw_y			output the dimension y of raw data.
	* \param raw_frame_id	output the frame id of current raw data
	*/
	int GetRawData(std::vector<short>& raw_data, int& raw_x, int& raw_y, int& raw_frame_id) {
		raw_data_mutex.lock();
		raw_frame_id = this->raw_frame_id;
		raw_x = this->raw_x;
		raw_y = this->raw_y;
		raw_data = this->raw_data;
		raw_data_mutex.unlock();
		return 1;
	}

	/**
	* Get raw mutual capacitive touch data.
	*
	* when a new frame arrives, event_raw_data will be set.
	*
	* \param raw_data		output the raw 16 bits data. 8 bits data is multplied by 8 before output.
	* \param raw_x			output the dimension x of raw data.
	* \param raw_y			output the dimension y of raw data.
	* \param raw_frame_id	output the frame id of current raw data
	* \param raw_timestamp	output the timestamp of current raw data
	* 	*/
	int GetRawData(std::vector<short>& raw_data, int& raw_x, int& raw_y, int& raw_frame_id, unsigned long long& timestamp_sys) {
		raw_data_mutex.lock();
		raw_frame_id = this->raw_frame_id;
		raw_x = this->raw_x;
		raw_y = this->raw_y;
		raw_data = this->raw_data;
		timestamp_sys = this->raw_timestamp_sys;
		raw_data_mutex.unlock();
		return 1;
	}

	/**
	* Get the touch points.
	* 
	* point format is TouchPoint. when a new touch point arrives, event_touch_data will be set.
	*/
	int GetTouchData(std::vector<TouchPoint>& touch_points) {
		touch_data_mutex.lock();

		touch_points.resize(touch_data.size());
		for (int i = 0; i < touch_data.size(); i++) {
			touch_points[i].timestamp_sys = touch_timestamp_sys[i];

			touch_points[i].touch_status = touch_data[i].tchstatus;
			touch_points[i].x_normalize = float(touch_data[i].x_pos) / touch_x_resolution;
			touch_points[i].y_normalize = float(touch_data[i].y_pos) / touch_y_resolution;

			touch_points[i].vect0 = (touch_data[i].vect & 0x0F) | (touch_data[i].vect & 0x08 ? 0xF0 : 0x00);
			touch_points[i].vect1 = (touch_data[i].vect >> 4) | (touch_data[i].vect & 0x80 ? 0xF0 : 0x00);

			touch_points[i].amplitude = touch_data[i].ampl;

			int areahw_exp = (touch_data[i].areahw0 >> 5) & 0x03;
			touch_points[i].area = (touch_data[i].areahw0 & 0x1F) << areahw_exp;
			touch_points[i].width_mm = ((touch_data[i].areahw1 >> 4) & 0x0F) << areahw_exp;
			touch_points[i].height_mm = (touch_data[i].areahw1 & 0x0F) << areahw_exp;
		}

		touch_data_mutex.unlock();

		return 1;
	}

	/**
	* Given touch status, extract type and event. Refer to Atmel 40002131A.pdf, pp.85
	* 
	* \param out_type:		0:	reserved							\n
	*						1:	finger								\n
	*						2:	passive stylus						\n
	*						5:	glove								\n
	*						6:	large touch							\n
	* \param out_event:		0:	no event							\n
	*						1:	move, touch position changed		\n
	*						2:	unsuppressed						\n
	*						3:	suppressed							\n
	*						4:	down, just touched					\n
	*						5:	up, just left						\n
	*						6:	unsupsup							\n
	*						7:	unsupup								\n
	*						8:	downsup								\n
	*						9:	downup								\n
	* \param in_tchstatus	the touch status byte
	* \return				whether touch detected
	*/
	static bool TouchStatus(int& out_type, int& out_event, uint8_t in_tchstatus) {
		out_type = (in_tchstatus & 0x70) >> 4;
		out_event = in_tchstatus & 0x0F;
		return in_tchstatus & 0x80;
	}

	/**
	* Calculate the angle and magnitude of each tilted touch point, given 2 vectors
	* 
	* \param angle_deg		out, the rotate angle around y axis (-90 ~ +90). 
	* \param magnitude		out, the tilt confidence
	* \param vect1			in, the vect1 component of the touch data
	* \param vect0			in, the vect0 component of the touch data
	*/
	static void TouchAngleMagnitude(float& angle_deg, float& magnitude, char vect1, char vect0) {
		angle_deg = atan2(vect1, vect0) * 180. / (3.14 * 2);
		magnitude = sqrt(vect1 * vect1 + vect0 * vect0);
	}

public:
	int								size_x_mm = 195;
	int								size_y_mm = 345;

	HANDLE							event_raw_data = nullptr;
	int								raw_bits = 8;		//	8 or 16
	int								raw_x = 0, raw_y = 0;
	int								raw_frame_id = 0;
	unsigned long long				raw_timestamp_sys = 0;
	std::vector<short>				raw_data;
	std::mutex						raw_data_mutex;

	HANDLE							event_touch_data = nullptr;
	unsigned short					touch_x_resolution = 0, touch_y_resolution = 0;
	std::vector<unsigned long long>	touch_timestamp_sys;
	std::vector<touch_status>		touch_data;
	std::mutex						touch_data_mutex;

protected:
	int WriteRegister(const uint8_t const* buf, int start_register, size_t count) {
		if (!handle) {
			std::cout << "error: MaxTouchDevice::WriteRegister(), handle is NULL" << std::endl;
			return 0;
		}

		size_t off = 0;

		while (off < count) {
			size_t bytes_remain = count - off;
			size_t bytes_transfer = bytes_remain <= MXT_HID_WRITE_DATA_SIZE ? bytes_remain : MXT_HID_WRITE_DATA_SIZE;

			hid_packet pkt;

			//	send request
			pkt.report_id = 0x01;
			pkt.cmd = HIDRAW_CMD_ID;
			pkt.rx_bytes = MXT_HID_ADDR_SIZE + bytes_transfer;
			pkt.tx_bytes = 0;
			pkt.address = start_register + off;
			memcpy(pkt.write_data, buf + off, bytes_transfer);

			WritePacket(pkt);

			//	read response
			ReadPacket(pkt);

			//	copy data
			if (pkt.result != 0x04) {
				std::cout << "response incorrect" << std::endl;

				std::cout << "the response packet is: " << std::hex
					<< int(pkt.report_id) << ' ' 
					<< int(pkt.result) << ' '
					<< int(pkt.bytes_read) << ', ';
				for (int i = 0; i < MXT_HID_READ_DATA_SIZE; i++)
					std::cout << int(pkt.read_data[i]) << ' ';
				std::cout << std::endl;
			}

			off += bytes_transfer;
		}

		return 1;
	}

	int ReadRegister(uint8_t* buf, int start_register, size_t count) {
		if (!handle) {
			std::cout << "error: MaxTouchDevice::ReadRegister(), handle is NULL" << std::endl;
			return 0;
		}

		size_t off = 0;

		while (off < count) {
			size_t bytes_remain = count - off;
			size_t bytes_transfer = bytes_remain <= MXT_HID_READ_DATA_SIZE ? bytes_remain : MXT_HID_READ_DATA_SIZE;

			hid_packet pkt;

			//	send request
			pkt.report_id = 0x01;
			pkt.cmd = HIDRAW_CMD_ID;
			pkt.rx_bytes = MXT_HID_ADDR_SIZE;
			pkt.tx_bytes = bytes_transfer;
			pkt.address = start_register + off;

			WritePacket(pkt);

			//	read response
			ReadPacket(pkt);

			//	copy data
			if (pkt.result != 0x00) {
				std::cout << "response incorrect" << std::endl;

				std::cout << "the response packet is: " << std::hex
					<< int(pkt.report_id) << ' '
					<< int(pkt.result) << ' '
					<< int(pkt.bytes_read) << ', ';
				for (int i = 0; i < MXT_HID_READ_DATA_SIZE; i++)
					std::cout << int(pkt.read_data[i]) << ' ';
				std::cout << std::endl;
			}
			else if (pkt.bytes_read != bytes_transfer)
				std::cout << "response bytes incorrect" << std::endl;
			else
				memcpy(buf + off, pkt.read_data, bytes_transfer);

			off += bytes_transfer;
		}

		return 1;
	}

	int WritePacket(struct hid_packet& write_pkt) {
		if (!handle) {
			std::cout << "error: MaxTouchDevice::WritePacket(), handle is NULL" << std::endl;
			return 0;
		}

		if (hid_write(handle, (unsigned char*)&write_pkt, sizeof(write_pkt)) < 0) {
			std::cout << "error: MaxTouchDevice::WritePacket() failed" << std::endl;
			return 0;
		}

		//std::cout << "write packet: " << int(write_pkt.report_id) << ' ';
		//for (int i = 0; i < MXT_HID_READ_DATA_SIZE + 2; i++)
		//	std::cout << int(write_pkt.raw_data[i]) << ' ';
		//std::cout << std::endl;

		return 1;
	}

	int ReadPacket(struct hid_packet& read_pkt) {
		if (!handle) {
			std::cout << "error: MaxTouchDevice::ReadPacket(), handle is NULL" << std::endl;
			return 0;
		}

		if (hid_read(handle, (unsigned char*)&read_pkt, sizeof(read_pkt)) < 0) {
			std::cout << "error: MaxTouchDevice::ReadPacket() failed" << std::endl;
			return 0;
		}

		//std::cout << "read packet: " << int(read_pkt.report_id) << ' ';
		//for (int i = 0; i < MXT_HID_READ_DATA_SIZE + 2; i++)
		//	std::cout << int(read_pkt.raw_data[i]) << ' ';
		//std::cout << std::endl;

		return 1;
	}

	int ClearHIDBuffer() {
		int pkt_cleared = 0;
		struct hid_packet pkt;

		do {
			int ret = hid_read_timeout(handle, (unsigned char*)&pkt, sizeof(pkt), 100);
			if (ret == 0)	//	no more packets
				break;
			else if (ret > 0) {
				//std::cout << "warning: MaxTouchDevice::ClearHIDBuffer(), clear packet" << pkt_cleared << std::endl;
				pkt_cleared++;
			}
			else {
				std::cout << "error: MaxTouchDevice::ClearHIDBuffer(), hid_read_timeout error" << std::endl;
				break;
			}

			//	more than 50 packets cleared, most likely the device is not closed correctly last time
			if (pkt_cleared > 50) {
				std::cout << "warning: MaxTouchDevice::ClearHIDBuffer(), auto return detected, stop them" << std::endl;

				pkt.report_id = 0x01;
				pkt.dbg_cmd = 0x00;
				WritePacket(pkt);

				pkt.report_id = 0x01;
				pkt.dbg_cmd = 0xE2;
				WritePacket(pkt);

				pkt_cleared = 0;
			}

		} while (true);

		return pkt_cleared;
	}

	int ReadInfoBlock() {
		//	==================== read id info ====================
		//	read header
		ReadRegister((uint8_t*)&info.id, 0x0000, sizeof(mxt_id_info));

		//	calculate whole size and read whole block
		int block_size = sizeof(mxt_id_info) +
			info.id.num_objects * sizeof(mxt_object) +
			sizeof(mxt_raw_crc);
		info.raw_info.resize(block_size);
		ReadRegister((uint8_t*)&info.raw_info[0], 0x0000, info.raw_info.size());

		//	copy to object and crc
		info.objects.resize(info.id.num_objects);
		memcpy(
			(uint8_t*)&info.objects[0],
			(uint8_t*)&info.raw_info[0] + sizeof(mxt_id_info),
			info.id.num_objects * sizeof(mxt_object)
		);

		size_t crc_area_size = sizeof(struct mxt_id_info) + info.id.num_objects * sizeof(struct mxt_object);
		info.crc = convert_crc(*(mxt_raw_crc*)((uint8_t*)&info.raw_info[0] + crc_area_size));

		//	check crc
		uint32_t calc_crc = mxt_calculate_crc((uint8_t*)&info.raw_info[0], crc_area_size);
		if (calc_crc == 0) {
			std::cout << "error: MaxTouchDevice::ReadInfoBlock(), zero crc" << std::endl;
		}
		else if (calc_crc != info.crc) {
			std::cout << "error: MaxTouchDevice::ReadInfoBlock(), crc not match" << std::endl;
		}

		//	==================== calculate report_id_map from object ====================
		info.max_report_id = 1;

		for (int i = 0; i < info.objects.size(); i++) {
			mxt_object& obj = info.objects[i];
			info.max_report_id += (obj.instances_minus_one + 1) * obj.num_report_ids;
		}

		report_id_map.resize(info.max_report_id);

		int report_id_count = 1;
		for (int i = 0; i < info.objects.size(); i++) {
			mxt_object& obj = info.objects[i];

			for (int instance = 0; instance < obj.instances_minus_one + 1; instance++) {
				for (int report_index = 0; report_index < obj.num_report_ids; report_index++) {
					report_id_map[report_id_count].object_type = obj.type;
					report_id_map[report_id_count].instance = instance;
					report_id_count++;
				}
			}
		}

		return 1;
	}

	int InitAutoReturn(bool diag_flip = false) {
		//	==================== Init auto return touch data ====================
		if (!touch_data.empty()) {
			{	//	set T100 object, set touch report
				int t100_addr = mxt_get_object_address(TOUCH_MULTITOUCHSCREEN_T100, 0);
				int diag_cmd_addr = t100_addr + MXT_T100_TCHAUX_OFFSET;
				uint8_t tchaux = 0x23;
				WriteRegister(&tchaux, diag_cmd_addr, 1);
			}

			{	//	set T100 object, set number of touch points
				int t100_addr = mxt_get_object_address(TOUCH_MULTITOUCHSCREEN_T100, 0);
				int diag_cmd_addr = t100_addr + MXT_T100_NUMTCH_OFFSET;
				uint8_t num_touches = touch_data.size();
				WriteRegister(&num_touches, diag_cmd_addr, 1);
			}

			{	//	set T100 object, do not switch xy
				int t100_addr = mxt_get_object_address(TOUCH_MULTITOUCHSCREEN_T100, 0);
				int diag_cmd_addr = t100_addr + MXT_T100_CFG1_OFFSET;
				uint8_t cfg1 = touch_data.size();
				ReadRegister(&cfg1, diag_cmd_addr, 1);

				cfg1 &= 0x1F;
				if (diag_flip)
					cfg1 |= 0xE0;
				WriteRegister(&cfg1, diag_cmd_addr, 1);
			}

			{	//	get T100 object, x y resolution
				int t100_addr = mxt_get_object_address(TOUCH_MULTITOUCHSCREEN_T100, 0);
				int diag_cmd_addr = t100_addr + MXT_T100_XRANGE_OFFSET;
				ReadRegister((uint8_t*)&touch_x_resolution, diag_cmd_addr, 2);

				diag_cmd_addr = t100_addr + MXT_T100_YRANGE_OFFSET;
				ReadRegister((uint8_t*)&touch_y_resolution, diag_cmd_addr, 2);
			}
		}

		//	==================== Init debug interface ====================
		if (raw_bits) {
			{	//	set T6 object, output delta in debug interface
				int t6_addr = mxt_get_object_address(GEN_COMMANDPROCESSOR_T6, 0);
				int diag_cmd_addr = t6_addr + MXT_T6_DEBUGCTRL_OFFSET;
				uint8_t DELTAS8EN_cmd = raw_bits == 8 ? MXT_T6_DEBUGCTRL_DELTAS8EN : MXT_T6_DEBUGCTRL_DELTASEN;
				WriteRegister(&DELTAS8EN_cmd, diag_cmd_addr, 1);
			}

			{	//	set T100 object, output full screen raw data
				int t100_addr = mxt_get_object_address(TOUCH_MULTITOUCHSCREEN_T100, 0);
				int diag_cmd_addr = t100_addr + MXT_T100_DBGXOFFSET_OFFSET;
				uint8_t DBG_XY[4] = { 0, 0, 0, 0 };
				WriteRegister(DBG_XY, diag_cmd_addr, 4);
			}
		}

		//	==================== Start ====================
		hid_packet pkt;

		//	start auto return
		if (!touch_data.empty()) {
			pkt.report_id = 0x01;
			pkt.dbg_cmd = 0x88;
			WritePacket(pkt);

			ReadPacket(pkt);
			if (pkt.dbg_cmd != 0x88) {
				std::cout << "error MaxTouchDevice::InitAutoReturn(), read cmd: 0x"
					<< std::hex << pkt.dbg_cmd << " is not 0x88" << std::endl;
			}
		}

		//	start debug monitoring
		if (raw_bits) {
			pkt.report_id = 0x02;
			pkt.dbg_cmd = 0xE1;
			WritePacket(pkt);

			ReadPacket(pkt);
			if (pkt.dbg_cmd != 0xE1) {
				std::cout << "error MaxTouchDevice::InitAutoReturn(), read cmd: 0x"
					<< std::hex << pkt.dbg_cmd << " is not 0xE1" << std::endl;
			}
		}

		return 1;
	}

	void QueryThread() {
		int raw_debug_data_size = info.id.matrix_x_size * info.id.matrix_y_size * (raw_bits / 8) + 2;
		std::vector<unsigned char> debug_data_buffer;
		debug_data_buffer.resize(raw_debug_data_size);	//	max size of debug data buffer, according to datasheet

		raw_frame_id = 0;

		hid_packet pkt;
		bool stop_debug_responsed = false;
		const int pkt_data_size = MXT_HID_READ_DATA_SIZE - 1;
		std::vector<short>	tmp_raw_data;

		int t100_start_report_id = -1;
		for (int i = 0; i < report_id_map.size(); i++) {
			if (report_id_map[i].object_type == 100) {
				t100_start_report_id = i;
				break;
			}
		}

		//	we need to wait until stop debug get response
		while (query_flag || !stop_debug_responsed) {
			ReadPacket(pkt);

			if (!query_flag) {
				//	debug interface will send response, while auto return does not response
				if (raw_bits && pkt.dbg_cmd == 0xE2) {	
					stop_debug_responsed = true;
					break;
				}
				else if (!raw_bits) {
					break;
				}
			}

			//	==================== recieved auto return touch message ====================
			if (pkt.id_bytes == 0x00FA && t100_start_report_id >= 0) {
				if (pkt.msg_id < report_id_map.size()) {
					uint16_t	obj_id = report_id_map[pkt.msg_id].object_type;
					int			report_idx = pkt.msg_id - t100_start_report_id;
					int			touch_idx = report_idx - 2;
					if (touch_idx >= 0 && touch_idx < touch_data.size()) {
						LARGE_INTEGER t_now;
						QueryPerformanceCounter(&t_now);

						touch_data_mutex.lock();
						memcpy(&touch_data[touch_idx], pkt.msg_data, sizeof(touch_status));
						touch_timestamp_sys[touch_idx] = t_now.QuadPart * 100;
						touch_data_mutex.unlock();

						if (event_touch_data)
							SetEvent(event_touch_data);
					}
				}
				else {
					std::cout << "error: MaxTouchDevice::QueryThread(), pkt.msg_id >= report_id_map.size()" << std::endl;
				}

				//std::cout << "dbg_pkt: " << std::dec
				//	<< int(pkt.dbg_packet_num) << ", "
				//	<< int(pkt.dbg_num_packets) << ", "
				//	<< int(pkt.dbg_frame_num) << std::endl;

				//for (int i = 0; i < MXT_HID_READ_DATA_SIZE - 1; i++) {
				//	std::cout << std::dec << int(pkt.dbg_data[i]) << ' ';
				//}
				//std::cout << std::endl;
			}
			//	==================== recieved debug message for raw data ====================
			else if (raw_bits && pkt.dbg_num_packets) {
				int curr_pkt_offset = pkt_data_size * (pkt.dbg_packet_num - 1);
				int curr_pkt_data_size = pkt_data_size;
				if (pkt.dbg_packet_num == pkt.dbg_num_packets) {
					curr_pkt_data_size = raw_debug_data_size - curr_pkt_offset;
				}
				memcpy(&debug_data_buffer[curr_pkt_offset], pkt.dbg_data, curr_pkt_data_size);

				//	the last debug packet, copy data
				if (pkt.dbg_packet_num == pkt.dbg_num_packets) {
					if (debug_data_buffer.front() != debug_data_buffer.back()) {
						std::cout << "error: MaxTouchDevice::QueryThread(), debug data frame id not match" << std::endl;
					}

					LARGE_INTEGER t_now;
					QueryPerformanceCounter(&t_now);

					raw_data_mutex.lock();
					raw_frame_id++;
					raw_timestamp_sys = t_now.QuadPart * 100;
					raw_x = info.id.matrix_x_size;
					raw_y = info.id.matrix_y_size;
					raw_data.resize(raw_x * raw_y);
					if (raw_bits == 8) {
						for (int i = 0; i < raw_x * raw_y; i++)
							raw_data[i] = char(debug_data_buffer[i + 1]) * 8;
					}
					else if (raw_bits == 16) {
						memcpy(&raw_data[0], &debug_data_buffer[1], raw_x * raw_y * 2);
					}
					if (diag_flip_flag) {
						//	flip raw data
						int flip_raw_x = raw_y;
						int flip_raw_y = raw_x;

						tmp_raw_data = raw_data;
						for (int y = 0; y < flip_raw_y; y++) {
							for (int x = 0; x < flip_raw_x; x++) {
								int old_x = raw_x - y - 1;
								int old_y = raw_y - x - 1;
								raw_data[y * flip_raw_x + x] = tmp_raw_data[old_x * raw_y + old_y];
							}
						}

						raw_x = flip_raw_x;
						raw_y = flip_raw_y;
					}
					raw_data_mutex.unlock();

					if (event_raw_data)
						SetEvent(event_raw_data);
				}
			}
			//	========================================
		}

		terminate_flag = true;
	}

	void PrintInfoBlock() {
		std::cout << std::dec << "info.id: \n"
			<< "\tfamily:\t\t" << int(info.id.family) << '\n'
			<< "\tvariant:\t" << int(info.id.variant) << '\n'
			<< "\tversion:\t" << int(info.id.version) << '\n'
			<< "\tbuild:\t\t" << int(info.id.build) << '\n'
			<< "\tmatrix_x_size:\t" << int(info.id.matrix_x_size) << '\n'
			<< "\tmatrix_y_size:\t" << int(info.id.matrix_y_size) << '\n'
			<< "\tnum_objects:\t" << int(info.id.num_objects) << std::endl;

		for (int i = 0; i < info.objects.size(); i++) {
			std::cout << "object " << i << ":\n"
				<< "\ttype:\t\t\t" << int(info.objects[i].type) << '\n'
				<< "\tstart_pos:\t\t" << *(unsigned short*)&info.objects[i].start_pos_lsb << '\n'
				<< "\tsize_minus_one:\t\t" << int(info.objects[i].size_minus_one) << '\n'
				<< "\tinstances_minus_one:\t" << int(info.objects[i].instances_minus_one) << '\n'
				<< "\tnum_report_ids:\t\t" << int(info.objects[i].num_report_ids) << std::endl;
		}

		for (int i = 0; i < report_id_map.size(); i++) {
			std::cout << std::dec << "report " << i << ": "
				<< int(report_id_map[i].object_type) << ", "
				<< int(report_id_map[i].instance) << std::endl;
		}
	}

	static void WaitUS(int wait_us) {
		LARGE_INTEGER frequency;	///<	frequency of the counter
		LARGE_INTEGER t_start;		///<	start time
		LARGE_INTEGER t_now;		///<	end time

		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&t_start);
		while (true) {
			QueryPerformanceCounter(&t_now);
			double elapsed_ms = double(t_now.QuadPart - t_start.QuadPart) * 1000000 / frequency.QuadPart;
			if (elapsed_ms >= wait_us)
				break;
		}
	}

protected:
	mxt_info						info;
	std::vector<mxt_report_id_map>	report_id_map;

	bool							diag_flip_flag		= false;	
	bool							query_flag			= false;
	bool							terminate_flag		= true;


protected:	//	T37 interface
	void InitT37Delta(struct t37_ctx& ctx) {
		int t6_addr = mxt_get_object_address(GEN_COMMANDPROCESSOR_T6, 0);

		/* T37 command address */
		ctx.diag_cmd_addr = t6_addr + MXT_T6_DIAGNOSTIC_OFFSET;

		/* Obtain Debug Diagnostic object's address */
		ctx.t37_addr = mxt_get_object_address(DEBUG_DIAGNOSTIC_T37, 0);

		/* Obtain Debug Diagnostic object's size */
		ctx.t37_size = mxt_get_object_size(DEBUG_DIAGNOSTIC_T37);

		ctx.t111_instances = mxt_get_object_instances(SPT_SELFCAPCONFIG_T111);

		ctx.t107_instances = mxt_get_object_instances(PROCI_ACTIVESTYLUS_T107);

		ctx.t100_instances = mxt_get_object_instances(TOUCH_MULTITOUCHSCREEN_T100);

		ctx.t9_instances = mxt_get_object_instances(TOUCH_MULTITOUCHSCREEN_T9);

		/* Minus header */
		ctx.page_size = ctx.t37_size - 2;

		//	delta mode config data
		ctx.x_size = info.id.matrix_x_size;
		ctx.y_size = info.id.matrix_y_size;
		ctx.data_values = ctx.x_size * ctx.y_size;
		ctx.passes = 1;
		ctx.pages_per_pass = (ctx.data_values * 2 + (ctx.page_size - 1)) / ctx.page_size;
		ctx.stripe_width = ctx.y_size;

		/* allocate t37 buffers */
		ctx.t37_buf = (t37_diagnostic_data*)calloc(1, ctx.t37_size);
		if (!ctx.t37_buf) {
			std::cout << "error: InitT37Delta(), alloc t37_buf failed" << std::endl;
		}

		/* allocate data buffer */
		ctx.data_buf.resize(ctx.data_values);

		ctx.instance = 0;

		//	print data
		std::cout << "ctx config:\n"
			<< "\tctx.x_size:\t\t" << ctx.x_size << '\n'
			<< "\tctx.y_size:\t\t" << ctx.y_size << '\n'
			<< "\tctx.data_values:\t" << ctx.data_values << '\n'
			<< "\tctx.passes:\t\t" << ctx.passes << '\n'
			<< "\tctx.pages_per_pass:\t" << ctx.pages_per_pass << '\n'
			<< "\tctx.stripe_width:\t" << ctx.stripe_width << std::endl;
	}

	void QueryT37DeltaThread(struct t37_ctx& ctx) {
		int frame_id = 0;

		while (query_flag) {
			/* iterate through stripes */
			/* Calculate stripe parameters */
			ctx.stripe_starty = 0;
			ctx.stripe_endy = ctx.stripe_starty + ctx.stripe_width - 1;
			ctx.x_ptr = 0;
			ctx.y_ptr = ctx.stripe_starty;
			ctx.pass = 0;

			for (ctx.page = 0; ctx.page < ctx.pages_per_pass; ctx.page++) {
				if (!mxt_get_t37_page(&ctx)) {
					terminate_flag = true;
					return;
				}

				mxt_debug_insert_data(&ctx);
			}

			//	copy data
			raw_data_mutex.lock();
			raw_frame_id = frame_id;
			raw_x = ctx.x_size;
			raw_y = ctx.y_size;
			raw_data.assign(ctx.data_buf.begin(), ctx.data_buf.end());
			raw_data_mutex.unlock();

			if (event_raw_data)
				SetEvent(event_raw_data);

			frame_id++;
		}

		terminate_flag = true;
	}

protected:	//	functions copied (with modification) from max-app
	uint16_t mxt_get_object_address(uint16_t object_type, uint8_t instance) {
		for (int i = 0; i < info.objects.size(); i++) {
			mxt_object& obj = info.objects[i];

			/* Does object type match? */
			if (obj.type == object_type) {
				/* Are there enough instances defined in the firmware? */
				if (obj.instances_minus_one >= instance) {
					return mxt_get_start_position(obj, instance);
				}
				else {
					std::cout << "error: MaxTouchDevice::mxt_get_object_address, instance not present on device" << std::endl;
					return 0;
				}
			}
		}

		std::cout << "error: MaxTouchDevice::mxt_get_object_address, object not present on device" << std::endl;
		return 0;
	}

	uint8_t mxt_get_object_size(uint16_t object_type) {
		int i = mxt_get_object_table_num(object_type);
		if (i == 255) {
			return OBJECT_NOT_FOUND;
		}

		return info.objects[i].size_minus_one + 1;
	}

	uint8_t mxt_get_object_table_num(uint16_t object_type) {
		for (int i = 0; i < info.objects.size(); i++) {
			if (info.objects[i].type == object_type) {
				return i;
			}
		}
		return 255;
	}

	uint8_t mxt_get_object_instances(uint16_t object_type) {
		for (int i = 0; i < info.objects.size(); i++) {
			if (info.objects[i].type == object_type) {
				return info.objects[i].instances_minus_one + 1;
			}
		}

		return 0;
	}

	int mxt_get_t37_page(struct t37_ctx* ctx) {
		if (ctx->pass == 0 && ctx->page == 0) {
			WriteRegister(&ctx->mode, ctx->diag_cmd_addr, 1);
		}
		else {
			uint8_t page_up_cmd = PAGE_UP;
			WriteRegister(&page_up_cmd, ctx->diag_cmd_addr, 1);
		}

		/* Read back diagnostic register in T6 command processor until it has been
		 * cleared. This means that the chip has actioned the command */
		int		failures = 0;
		uint8_t read_command = 1;
		while (read_command) {
			WaitUS(500);
			ReadRegister(&read_command, ctx->diag_cmd_addr, 1);

			if (read_command) {
				failures++;

				if (failures > 500) {
					std::cout << "mxt_get_t37_page, Timeout waiting for command to be actioned" << std::endl;
					return 0;
				}
			}
		}
		ReadRegister((uint8_t*)ctx->t37_buf, ctx->t37_addr, ctx->t37_size);

		//std::cout << "read, mode: " << int(ctx->t37_buf->mode)
		//	<< ", page: " << int(ctx->t37_buf->page) << std::endl;

		if (ctx->t37_buf->mode != ctx->mode) {
			std::cout << "error: mxt_get_t37_page(), Bad mode in diagnostic data read" << std::endl;
			return 0;
		}

		if (ctx->t37_buf->page != (ctx->pages_per_pass * ctx->pass + ctx->page)) {
			std::cout << "error: mxt_get_t37_page(), Bad page in diagnostic data read" << std::endl;
			return 0;
		}

		return 1;
	}

	int mxt_debug_insert_data(struct t37_ctx* ctx) {
		for (int i = 0; i < ctx->page_size; i += 2) {

			uint16_t value = (ctx->t37_buf->data[i + 1] << 8) | ctx->t37_buf->data[i];

			int ofs = ctx->y_ptr;

			/* The last page may overlap the end of the matrix */
			if (ofs >= ctx->data_values)
				return 1;

			ctx->data_buf[ofs] = value;

			ctx->y_ptr++;
		}

		return 1;
	}

	static uint32_t convert_crc(const struct mxt_raw_crc& crc)
	{
		return ((crc.CRC_hi << 16u) | (crc.CRC));
	}

	static uint32_t crc24(uint32_t crc, uint8_t firstbyte, uint8_t secondbyte) {
		static const uint32_t CRCPOLY = 0x0080001B;
		uint32_t result;
		uint16_t data_word;

		data_word = (uint16_t)((uint16_t)(secondbyte << 8u) | firstbyte);
		result = ((crc << 1u) ^ (uint32_t)data_word);

		/* Check if 25th bit is set, and XOR the result to create 24-bit checksum */
		if (result & 0x1000000) {
			result ^= CRCPOLY;
		}
		return result;
	}

	static uint32_t mxt_calculate_crc(uint8_t* base_addr, size_t size) {
		static const uint32_t MASK_24_BITS = 0x00FFFFFF;
		uint32_t calc_crc = 0; /* Calculated checksum */
		uint16_t crc_byte_index = 0;

		/* Call the CRC function crc24() iteratively to calculate the CRC,
		 * passing it two bytes at a time.  */
		while (crc_byte_index < ((size % 2) ? (size - 1) : size)) {
			calc_crc = crc24(calc_crc, *(base_addr + crc_byte_index),
				*(base_addr + crc_byte_index + 1));
			crc_byte_index += 2;
		}

		/* Call crc24() for the final byte, plus an extra
		 *  0 value byte to make the sequence even if it's odd */
		if (size % 2) {
			calc_crc = crc24(calc_crc, *(base_addr + crc_byte_index), 0);
		}

		/* Mask 32-bit calculated checksum to 24-bit */
		calc_crc &= calc_crc & MASK_24_BITS;

		return calc_crc;
	}

	static uint16_t mxt_get_start_position(const mxt_object& obj, uint8_t instance)
	{
		return (obj.start_pos_msb * 256) + obj.start_pos_lsb
			+ ((obj.size_minus_one + 1) * instance);
	}

};


#endif	//	MAXTOUCH_H
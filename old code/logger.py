import serial
import struct
import time
import numpy as np
import csv

class VitalSign:
    # Inisialisasi variabel
    outHeartNew_CM = 0
    thresh_HeartCM = 0.4
    thresh_diffEst = 15
    LENGTH_BUFFER = 40
    heartRateEstDisplay_CircBuffer = 48 * np.ones(LENGTH_BUFFER)
    alpha = 0.5
    seconds_since_epoch = time.time()
    time_struct = time.localtime(seconds_since_epoch)
    date_time_yyyymmddhhmm = f"{time_struct.tm_year}{time_struct.tm_mon:02}{time_struct.tm_mday:02}{time_struct.tm_hour:02}{time_struct.tm_min:02}"
    filename = date_time_yyyymmddhhmm + ".csv"

    def __init__(self, command_port, data_port, command_baudrate=115200, data_baudrate=912600):
        # Inisialisasi kedua port serial
        self.ser_command = serial.Serial(command_port, command_baudrate, timeout=1)
        self.ser_data = serial.Serial(data_port, data_baudrate, timeout=1)
        self.buffer_data = b''  # Buffer untuk menyimpan data
        self.tlv_header = 0  # header code frame data
        self.data_len = 0  # Panjang frame data

        self.config_commands = [
            "sensorStop\n",
            "flushCfg\n",
            "dfeDataOutputMode 1\n",
            "channelCfg 15 3 0\n",
            "adcCfg 2 1\n",
            "adcbufCfg 0 1 0 1\n",
            "profileCfg 0 77 7 6 57 0 0 70 1 200 4000 0 0 48\n",
            "chirpCfg 0 0 0 0 0 0 0 1\n",
            "frameCfg 0 0 2 0 20 1 0\n",
            "guiMonitor 0 0 0 0 1\n",
            "vitalSignsCfg 0.3 1.0 256 512 4 0.1 0.05 100000 100000\n",
            "motionDetection 1 20 3.0 0\n",
            "sensorStart\n\n\n\n"
        ]
        self.config_index = 0  # Indeks perintah yang sedang dikirim
        self.start_sensing_bool = True  # Flag untuk memulai pengiriman

        # Inisialisasi file CSV dan tuliskan header
        self.csv_file = open(self.filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp',
                                  'outputFilterHeartOut',
                                  'final_heart_rate', 
                                  'heart_rate_est_peak', 
                                  'heart_rate_est_fft',
                                  'heart_rate_est_fft_4hz',
                                  'heartRateEst_xCorr', 
                                  'confidence_heart', 
                                  'range_bin_value',
                                  'rangeBinPhaseIndex',
                                  'unwrapPhasePeak_mm',
                                  'sumEnergyHeartWfm'
                                  ])

        # self.csv_writer.writerow([
        #     'Timestamp', 
        #     'range_bin_phase_index',
        #     'unwrapPhasePeak_mm'
        #     ])

    def send_config_commands(self):
        """
        Mengirimkan semua perintah konfigurasi ke perangkat.
        """
        while self.start_sensing_bool and self.config_index < len(self.config_commands):
            cmd = self.config_commands[self.config_index]
            self.ser_command.write(cmd.encode('utf-8'))  # Kirim perintah ke port command
            print(f"Sending command: {cmd.strip()}")
            self.config_index += 1
            time.sleep(0.1)  # Delay untuk memastikan perangkat menerima perintah

        # Setelah semua perintah dikirim, mulai membaca data
        self.start_sensing_bool = False

    # def onSerialDATAReceived(self):
    #     """
    #     Fungsi untuk menerima data serial dari perangkat.
    #     """
    #     try:
    #         # Membaca semua data yang tersedia dari port data
    #         data_new = self.ser_data.read(self.ser_data.in_waiting or 1)

    #         # Mengecek apakah data baru adalah permulaan frame
    #         if self.checkFirst8Bytes(data_new):
    #             self.buffer_data = data_new  # Mulai buffer baru
    #             # Membaca panjang data dari byte ke-12 sampai ke-15
    #             self.data_len = struct.unpack_from('<I', data_new, 12)[0]  # Little endian 4-byte unsigned int
    #         else:
    #             # Menambahkan data baru ke buffer yang sudah ada
    #             self.buffer_data += data_new

    #         # Mengecek apakah buffer data sudah mencapai panjang frame yang diharapkan
    #         if len(self.buffer_data) >= self.data_len:
    #             # Jika sudah, lakukan pengolahan data vital sign
    #             self.displayVitalSign(self.buffer_data)
    #     except serial.SerialException as e:
    #         print(f"Serial error: {e}")

    def onSerialDATAReceived(self):
        try:
            data_new = self.ser_data.read(self.ser_data.in_waiting or 1)
            # print(f"Received Data (Raw): {data_new.hex()}")  # Print data HEX
            

            if self.checkFirst8Bytes(data_new):
                self.buffer_data = data_new
                self.tlv_header = struct.unpack_from('<I', data_new, 40)[0]
                self.data_len = struct.unpack_from('<I', data_new, 44)[0]             

            else:
                self.buffer_data += data_new
                

            if len(self.buffer_data) >= self.data_len:
                # print(f"Detected New Packet - Header: {self.tlv_header}")
                # print(f"Detected New Packet - Length: {self.data_len}")
                # print(f"Complete Frame Received: {len(self.buffer_data)} bytes")
                self.displayVitalSign(self.buffer_data)

                
                self.buffer_data = b''

        except serial.SerialException as e:
            print(f"Serial error: {e}")


    def checkFirst8Bytes(self, data):
        """
        Mengecek apakah 8 byte pertama sesuai dengan ketentuan.
        """
        if len(data) >= 8:
            # Mengecek apakah 8 byte pertama sesuai dengan pola yang ditentukan
            return (data[0] == 0x02 and data[1] == 0x01 and
                    data[2] == 0x04 and data[3] == 0x03 and
                    data[4] == 0x06 and data[5] == 0x05 and
                    data[6] == 0x08 and data[7] == 0x07)
        return False

    def displayVitalSign(self, data):
        """
        Mengolah data vital sign dari buffer lengkap.
        """
        # Membaca nilai-nilai vital sign dari buffer
        range_bin_value = struct.unpack_from('<f', data, 52)[0]  
        outputFilterHeartOut = struct.unpack_from('<f', data, 72)[0]  
        heart_rate_est_fft = struct.unpack_from('<f', data, 76)[0]   
        heart_rate_est_fft_4hz = struct.unpack_from('<f', data, 80)[0]   
        heartRateEst_xCorr = struct.unpack_from('<f', data, 84)[0]   
        heart_rate_est_peak = struct.unpack_from('<f', data, 92)[0]  
        confidence_heart = struct.unpack_from('<f', data, 104)[0]    
        sumEnergyHeartWfm = struct.unpack_from('<f', data, 116)[0]    
        rangeBinPhaseIndex = struct.unpack_from('<I', data, 50)[0]    
        unwrapPhasePeak_mm = struct.unpack_from('<f', data, 64)[0]    


        # Hitung estimasi heart rate final
        final_heart_rate = self.calculate_heart_rate_est_display(confidence_heart, heart_rate_est_fft, heart_rate_est_peak)

        # Dapatkan timestamp saat ini
        timestamp = time.time()

        # Tulis log ke file CSV
        self.csv_writer.writerow([timestamp,
                                  outputFilterHeartOut,
                                  final_heart_rate, 
                                  heart_rate_est_peak, 
                                  heart_rate_est_fft,
                                  heart_rate_est_fft_4hz,
                                  heartRateEst_xCorr, 
                                  confidence_heart, 
                                  range_bin_value,
                                  rangeBinPhaseIndex,
                                  unwrapPhasePeak_mm,
                                  sumEnergyHeartWfm
                                ])
        # self.csv_writer.writerow([timestamp,
        #                           rangeBinPhaseIndex,
        #                           unwrapPhasePeak_mm
        #                           ])

        # Tampilkan atau olah hasil
        # print(f"Heart Rate (Est. Final): {final_heart_rate} bpm")
        # print(f"Heart Rate (Est. Peak): {heart_rate_est_peak} bpm")
        # print(f"Heart Rate (FFT): {heart_rate_est_fft} bpm")
        # print(f"Confidence (Heart): {confidence_heart}")
        # print(f"Range Bin Value: {range_bin_value}")

        # print(f"Heart Rate (Est. Final): {final_heart_rate} bpm, Heart Rate (Est. Peak): {heart_rate_est_peak} bpm, Heart Rate (FFT): {heart_rate_est_fft} bpm, Confidence (Heart): {confidence_heart}, Range Bin Value: {range_bin_value}")

        # Reset buffer setelah pemrosesan
        self.buffer_data = b''

    def calculate_heart_rate_est_display(self, outConfidenceMetric_Heart, heartRateEstFFT, heartRateEstPeak):
        outHeartPrev_CM = self.outHeartNew_CM
        self.outHeartNew_CM = self.alpha * outConfidenceMetric_Heart + (1 - self.alpha) * outHeartPrev_CM

        # Penanganan NaN atau Inf pada CM
        if np.isnan(self.outHeartNew_CM) or np.isinf(self.outHeartNew_CM):
            self.outHeartNew_CM = 99

        # Hitung perbedaan estimasi heart rate
        diffEst_heartRate = abs(heartRateEstFFT - heartRateEstPeak)
        
        # Pilih heartRateEstDisplay berdasarkan kondisi
        if (self.outHeartNew_CM > self.thresh_HeartCM) or (diffEst_heartRate < self.thresh_diffEst):
            heartRateEstDisplay = heartRateEstFFT
        else:
            heartRateEstDisplay = heartRateEstPeak

        # Update buffer circular heart rate
        self.heartRateEstDisplay_CircBuffer = np.roll(self.heartRateEstDisplay_CircBuffer, -1)
        self.heartRateEstDisplay_CircBuffer[-1] = heartRateEstDisplay

        # Hitung median dari buffer circular
        heartRateEstDisplayFinal = np.median(self.heartRateEstDisplay_CircBuffer)

        return heartRateEstDisplayFinal

    def close(self):
        """
        Menutup port serial dengan aman dan tutup file CSV.
        """
        if self.ser_command.is_open:
            self.ser_command.close()
        if self.ser_data.is_open:
            self.ser_data.close()
        # Tutup file CSV setelah selesai
        self.csv_file.close()

# Penggunaan:
# Inisialisasi port serial dan mulai mengirimkan perintah
if __name__ == "__main__":
    vital_sign = VitalSign('COM9', 'COM8')  # Sesuaikan dengan port serial Anda

    try:
        # Kirimkan perintah konfigurasi
        vital_sign.send_config_commands()

        # Setelah mengirim perintah, mulai membaca data
        while True:
            vital_sign.onSerialDATAReceived()
    except KeyboardInterrupt:
         print("Program dihentikan.")
    finally:
        vital_sign.close()

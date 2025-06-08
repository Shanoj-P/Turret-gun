import serial
import json

# Set your serial port here
SERIAL_PORT = '/dev/ttyUSB0'  # Replace with your correct serial port
BAUD_RATE = 115200
calib_data = {}

def save_calibration_data(data):
    """Save the calibration data to a JSON file."""
    with open("bno055_calibration.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Calibration data saved to bno055_calibration.json")

# Open serial port
with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:  # Increased timeout
    print("Listening for calibration data from Arduino...")

    while True:
        line = ser.readline().decode('utf-8').strip()

        if line:
            print(f"Received: {line}")  # Print the incoming data

            # Check if the line contains calibration data
            if line.startswith("accel_offset"):
                try:
                    # Parse the calibration data and store it in a dictionary
                    key, value = line.split(":")
                    calib_data[key] = int(value)
                    print(f"Parsed {key}: {value}")
                except ValueError:
                    pass

            if line == "CALIBRATION -> Sys:3 Gyro:3 Accel:3 Mag:3":  # This shows full calibration
                print("Full calibration achieved.")

        # If 's' is pressed in the terminal, save the data
        if input() == 's':  # Wait for user input
            save_calibration_data(calib_data)
            break  # Exit after saving

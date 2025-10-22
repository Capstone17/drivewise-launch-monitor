import smbus2 as smbus
import time
from collections import deque
import os
from datetime import datetime
import argparse

# INA219 Register Addresses
_REG_CONFIG = 0x00
_REG_SHUNTVOLTAGE = 0x01
_REG_BUSVOLTAGE = 0x02
_REG_POWER = 0x03
_REG_CURRENT = 0x04
_REG_CALIBRATION = 0x05

# Configurable Constants
I2C_BUS = 1
I2C_ADDRESS = 0x41
SAMPLE_INTERVAL = 2  # Data sampling interval in seconds
BATTERY_CAPACITY_WH = 100  # UPS battery capacity in watt-hours

class INA219:
    """Class to interface with the INA219 sensor for voltage, current, and power readings."""

    def __init__(self, i2c_bus=I2C_BUS, addr=I2C_ADDRESS, shunt_resistance=0.1):
        """
        Initializes the INA219 with default calibration for 32V and 2A range.

        Parameters:
        i2c_bus (int): The I2C bus number.
        addr (int): The I2C address of the INA219.
        shunt_resistance (float): Shunt resistor value in ohms.
        """
        self.bus = smbus.SMBus(i2c_bus)
        self.addr = addr
        self.shunt_resistance = shunt_resistance
        self._current_lsb = 0.1  # Current LSB = 100uA per bit
        self._power_lsb = 0.002  # Power LSB = 2mW per bit
        self.set_calibration_32V_2A()

    def write(self, address, data):
        """Writes a 16-bit value to a register on the INA219 sensor."""
        temp = [data >> 8, data & 0xFF]
        self.bus.write_i2c_block_data(self.addr, address, temp)

    def read(self, address):
        """Reads a 16-bit value from a register on the INA219 sensor."""
        data = self.bus.read_i2c_block_data(self.addr, address, 2)
        return (data[0] << 8) | data[1]

    def set_calibration_32V_2A(self):
        """Sets the INA219 to measure up to 32V and 2A."""
        self._cal_value = int(0.04096 / (self._current_lsb * self.shunt_resistance))
        self.write(_REG_CALIBRATION, self._cal_value)
        self.config = (0x2000 | 0x1800 | 0x07)  # 32V, 320mV gain, continuous mode
        self.write(_REG_CONFIG, self.config)

    def getShuntVoltage_mV(self):
        """Returns the shunt voltage in mV."""
        value = self.read(_REG_SHUNTVOLTAGE)
        return ((value - 65536) if value > 32767 else value) * 0.01
    
    def getBusVoltage_V(self):
        """Returns the bus voltage in V."""
        value = self.read(_REG_BUSVOLTAGE)
        return (value >> 3) * 0.004

    def getCurrent_mA(self):
        """Returns the current in mA."""
        value = self.read(_REG_CURRENT)
        return ((value - 65536) if value > 32767 else value) * self._current_lsb

    def getPower_W(self):
        """Returns the power in W."""
        value = self.read(_REG_POWER)
        return ((value - 65536) if value > 32767 else value) * self._power_lsb
    
    def getPercent(self, bus_voltage):
        """Calculates battery percentage based on bus voltage."""
        percent = ((bus_voltage - 9) / 3.6) * 100
        return min(max(percent, 0), 100)

    def estimate_remaining_time(self, current_power_draw):
        """
        Estimates the remaining time based on current power draw.

        Parameters:
        current_power_draw (float): Current power draw in W.

        Returns:
        float: Estimated remaining time in minutes.
        """
        if current_power_draw > 0:
            remaining_time_hours = BATTERY_CAPACITY_WH / current_power_draw
            return min(10000, remaining_time_hours * 60)  # Limits time to avoid impractical values
        return None


def return_battery_power():
    """
    Displays a formatted summary of key metrics with color highlights for easy readability.

    Parameters:
    bus_voltage (float): Voltage reading in V.
    current (float): Current reading in A.
    power (float): Power reading in W.
    percent (float): Battery percentage.
    remaining_time (float): Estimated remaining time in minutes.
    """

    try:
        # Create INA219 object
        ina219 = INA219()

        bus_voltage = ina219.getBusVoltage_V()
        current = ina219.getCurrent_mA() / 1000
        power = ina219.getPower_W()
        percent = ina219.getPercent(bus_voltage)
        remaining_time = ina219.estimate_remaining_time(power)


        # Format remaining time for better readability
        if remaining_time and remaining_time > 1440:  # Cap at 24 hours
            remaining_time_display = "More than 24 hrs"
        elif remaining_time and remaining_time > 60:
            hours = int(remaining_time // 60)
            minutes = int(remaining_time % 60)
            remaining_time_display = f"{hours} hrs {minutes} min"
        else:
            remaining_time_display = f"{remaining_time:.2f} min" if remaining_time else "Calculating..."

        print("'Load Voltage': bus_voltage, 'Current': current, 'Power': power, 'Percent': percent, 'Time': remaining_time_display")
    
    except IOError as e:
        print("I2C communication error:", e)
        return "Error"
    finally:
        # return output with power stage and remaining time
        return {'Load Voltage': bus_voltage, 'Current': current, 'Power': power, 'Percent': percent, 'Time': remaining_time_display}


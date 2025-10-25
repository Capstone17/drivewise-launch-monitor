# ---------------------------------------
#
# To shut off script temporarily: sudo systemctl stop webcamgolf.service
#
# ---------------------------------------


import logging

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service

from ble import (
    Advertisement,
    Characteristic,
    Service,
    Application,
    find_adapter,
    Descriptor,
    Agent,
)

import array
import sys
import subprocess
import json
import os
import glob

from video_ball_detector import process_video
from metrics.ruleBasedSystem import rule_based_system
from embedded.exposure_calibration import calibrate_exposure
from battery import return_battery_power
from auto_capture import *

MainLoop = None
try:
    from gi.repository import GLib

    MainLoop = GLib.MainLoop
except ImportError:
    import gobject as GObject

    MainLoop = GObject.MainLoop

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logHandler = logging.StreamHandler()
filelogHandler = logging.FileHandler("logs.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logHandler.setFormatter(formatter)
filelogHandler.setFormatter(formatter)
logger.addHandler(filelogHandler)
logger.addHandler(logHandler)

BaseUrl = "XXXXXXXXXXXX"

mainloop = None

BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
GATT_CHRC_IFACE =    'org.bluez.GattCharacteristic1'


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.NotSupported"


class NotPermittedException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.NotPermitted"


class InvalidValueLengthException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.InvalidValueLength"


class FailedException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.Failed"


def register_app_cb():
    logger.info("GATT application registered")


def register_app_error_cb(error):
    logger.critical("Failed to register application: " + str(error))
    mainloop.quit()


class rpiService(Service):
    """
    Dummy test service that provides characteristics and descriptors that
    exercise various API functionality.
    """

    rpi_SVC_UUID = "96f0284d-8895-4c08-baaf-402a2f7e8c5b"

    def __init__(self, bus, index):
        Service.__init__(self, bus, index, self.rpi_SVC_UUID, True)
        self.shared_data = {
            "metrics": None,
            "feedback": None
        }
        self.exposure = "200"
        self.add_characteristic(SwingAnalysisCharacteristic(bus, 0, self))
        self.add_characteristic(GenerateFeedbackCharacteristic(bus, 1, self))
        self.add_characteristic(FindIPCharacteristic(bus, 2, self))
        self.add_characteristic(CalibrationCharacteristic(bus, 3, self))
        self.add_characteristic(BatteryMonitorCharacteristic(bus, 4, self))
        # self.add_characteristic(PowerOffCharacteristic(bus, 3, self))


class SwingAnalysisCharacteristic(Characteristic):
    uuid = "d9c146d3-df83-49ec-801d-70494060d6d8"
    description = b"Start analysis and get results!"

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index, self.uuid, ["read", "write", "notify"], service,
        )
        self.notifying = False
        self.value = self.service.shared_data["metrics"] 
        self.add_descriptor(CharacteristicUserDescriptionDescriptor(bus, 0, self))

    def ReadValue(self, options):
        # Run computer vision script here
        # Run rule-based AI with all 3 json files to recieve 4 output metrics in dict + string message
        logger.debug("Analysis finished, sending metrics: " + repr(self.value))
        result_bytes = json.dumps(self.value).encode('utf-8')
        return [dbus.Byte(b) for b in result_bytes]
        
        
    def WriteValue(self, value, options):
        logger.debug("Received write command")
        
        # Create a config and manager classes necessary for auto capture
        low_config = LowRateDetectionConfig(
            low_fps=5.0,
            max_wait_seconds=120.0,
            score_threshold=0.20,
            min_consecutive_hits=2,
        )

        logger.debug("High speed capture config setup")
        high_config = HighSpeedCaptureConfig(
            duration_seconds=5.0,
            shutter_speed=200,
            width=196,
            height=128,
            framerate=550,
            camera_index=0,
        )

        logger.debug("AutoCapture Manager setup")
        manager = AutoCaptureManager(
            low_config=low_config,
            high_config=high_config,
            tail_frames_to_check=8,
            tail_stride=1,
            tail_score_threshold=0.20,
            tail_min_hits=2,
            max_high_attempts=5,
        )
        
        logger.debug("Beginning auto capture")
        capture_cycle = manager.acquire_clip(self.service.exposure)
        latest_file = str(capture_cycle.final_video)
        logger.info(
            "High-speed capture completed after %d attempt(s); final clip: %s",
            capture_cycle.attempts,
            latest_file,
        )
        if capture_cycle.detection_event:
            det_evt = capture_cycle.detection_event
            logger.debug(
                "Low-rate watcher trigger frame=%d score=%.3f source=%s",
                det_evt.frame_index,
                det_evt.score,
                det_evt.source,
            )
        if len(capture_cycle.all_videos) > 1:
            logger.debug(
                "Intermediate clips before final selection: %s",
                ", ".join(str(p) for p in capture_cycle.all_videos[:-1]),
            )


        # if self.service.exposure == None: 
        #     self.service.shared_data["metrics"] = {'face angle': 0, 'swing path': 0, 'attack angle': 0, 'side angle': 0}
        #     self.service.shared_data["feedback"] = "Please run calibration first!"
        # else:
        try:
            # Run video script
            subprocess.run(
                [
                    "./embedded/rpicam_run.sh",
                    "5s",  # Time in seconds
                    self.service.exposure
                ],
                check=True,
            )

            logger.info("processing video now")
            # Find most recent tst*.mp4 file in output directory
            # output_dir = os.path.expanduser("~/Documents/webcamGolf")
            # mp4_files = glob.glob(os.path.join(output_dir, "vid*.mp4"))
            # if not mp4_files:
            #     raise FileNotFoundError("No vid*.mp4 files found in webcamGolf directory")

            # latest_file = max(mp4_files, key=os.path.getmtime)
            # logger.info(f"Latest video file: {latest_file}")

            # For testing
            latest_file = "CEsticker_white_200exp1.mp4"

            # Process video
            result = process_video(
                latest_file,
                "ball_coords.json",
                "sticker_coords.json",
                "ball_frames",
            )
            if result != "skibidi":
                raise RuntimeError("Video processing did not complete")
            # Run metric calculations
            self.service.shared_data = rule_based_system("mid-iron")
            self.value = self.service.shared_data["metrics"]

        except subprocess.CalledProcessError as e:
            logger.error(f"Shell script failed: {e}")
            self.service.shared_data["metrics"] = {'face angle': 0, 'swing path': 0, 'attack angle': 0, 'side angle': 0}
            self.service.shared_data["feedback"] = "Script execution failed!"
            self.value = self.service.shared_data["metrics"]
            if self.notifying:
                self.notify_client()

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.service.shared_data["metrics"] = {'face angle': 0, 'swing path': 0, 'attack angle': 0, 'side angle': 0}
            self.service.shared_data["feedback"] = "Swing analysis failed! Please try again."
            self.value = self.service.shared_data["metrics"]
            if self.notifying:
                self.notify_client()

        else:
            # This block runs only if try block completes without exception
            logger.debug("Updated value after script")
            if self.notifying:
                self.notify_client()

        
    def StartNotify(self):
        if self.notifying:
            logger.debug("Already notifying")
            return
        logger.debug("StartNotify called")
        self.notifying = True
        # self.notify_client()

    def StopNotify(self):
        if not self.notifying:
            logger.debug("Not currently notifying")
            return
        logger.debug("StopNotify called")
        self.notifying = False

    def notify_client(self):
        if not self.notifying:
            logger.debug("Not notifying, skipping notify_client")
            return

        result_bytes = json.dumps(self.value).encode('utf-8')
        logger.debug("Emitting PropertiesChanged with updated value")
        self.PropertiesChanged(
        GATT_CHRC_IFACE,
        {"Value": [dbus.Byte(b) for b in result_bytes]},
        []
        )

class GenerateFeedbackCharacteristic(Characteristic):
    uuid = "2c58a217-0a9b-445f-adac-0b37bd8635c3"
    description = b"Generate feedback based on swing metrics!"

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index, self.uuid, ["read"], service,
        )
        self.value = self.service.shared_data["feedback"]
        self.add_descriptor(CharacteristicUserDescriptionDescriptor(bus, 1, self))
        logger.debug("Entered generatefeedback characteristic")


    def ReadValue(self, options):
        # take text from json file that has feedback
        self.value = self.service.shared_data["feedback"]
        logger.debug("sending feedback based on metrics: " + repr(self.value))
        result_bytes = json.dumps(self.value).encode('utf-8')
        return [dbus.Byte(b) for b in result_bytes]
    
class FindIPCharacteristic(Characteristic):
    uuid = "2c75511d-11b8-407d-b275-a295ef2c199f"
    description = b"Read to get IP!"

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index, self.uuid, ["read"], service,
        )
        self.value = "IP Failed"
        self.add_descriptor(CharacteristicUserDescriptionDescriptor(bus, 2, self))

    def ReadValue(self, options):
        # 
        self.value = subprocess.check_output(["hostname", "-I"], text=True, )
        logger.debug("Hostname found: " + repr(self.value))
        result_bytes = json.dumps(self.value).encode('utf-8')
        return [dbus.Byte(b) for b in result_bytes]
    
class CalibrationCharacteristic(Characteristic):
    uuid = "778c5d1a-315f-4baf-a23b-6429b84835e3"
    description = b"Use to calibrate the exposure of the camera!"

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index, self.uuid, ["write", "notify"], service,
        )
        self.notifying = False
        self.add_descriptor(CharacteristicUserDescriptionDescriptor(bus, 3, self))

    def WriteValue(self, value, options):
        logger.debug("received write command")
        try: 
            # Run calibration script
            logger.debug("Began calibration function")
            self.service.exposure = calibrate_exposure()
            logger.info(f"Exposure from calibration: {self.service.exposure}")
            # Run config script
            logger.debug("Calibration successful. Now running GS_config")
            subprocess.run(
                [
                    "./embedded/GS_config.sh",
                    # , If we want to specify width or height we should do so here
                    "224", # Width
                    "128"  # Height
                ],
                check=True,
            )
        except Exception as e:
            logger.error(f"Calibration function failed: {e}")
            if self.notifying:
                self.notify_client()

        except subprocess.CalledProcessError as e:
            logger.error(f"GS crop script failed: {e}")
            if self.notifying:
                self.notify_client()

        else:
            # This block runs only if try block completes without exception
            logger.debug("Calibration and GS_Crop successful")
            if self.notifying:
                self.notify_client()

    def StartNotify(self):
        if self.notifying:
            logger.debug("Already notifying")
            return
        logger.debug("StartNotify called")
        self.notifying = True
        # self.notify_client()

    def StopNotify(self):
        if not self.notifying:
            logger.debug("Not currently notifying")
            return
        logger.debug("StopNotify called")
        self.notifying = False

    def notify_client(self):
        if not self.notifying:
            logger.debug("Not notifying, skipping notify_client")
            return

        self.value = "Calibration complete!"
        result_bytes = json.dumps(self.value).encode('utf-8')
        logger.debug("Notifying values changed")
        self.PropertiesChanged(
        GATT_CHRC_IFACE,
        {"Value": [dbus.Byte(b) for b in result_bytes]},
        []
        )

class BatteryMonitorCharacteristic(Characteristic):
    uuid = "a834f0f7-89cc-453b-8be4-2905d27344bf"
    description = b"Regularly send the battery status to the app!"

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index, self.uuid, ["read","notify"], service,
        )
        self.notifying = False
        self.add_descriptor(CharacteristicUserDescriptionDescriptor(bus, 4, self))

    def ReadValue(self, options):
        # 
        self.value = return_battery_power()
        logger.debug("reading battery power: " + repr(self.value))
        result_bytes = json.dumps(self.value).encode('utf-8')
        return [dbus.Byte(b) for b in result_bytes]

    def StartNotify(self):
        if self.notifying:
            logger.debug("Already notifying")
            return
        logger.debug("StartNotify called")
        self.notifying = True

        self.value = return_battery_power()
        self.notify_client()

        # start periodic updates every 5 seconds
        self.notify_timer = GLib.timeout_add_seconds(60, self.check_battery)

    def StopNotify(self):
        if not self.notifying:
            logger.debug("Not currently notifying")
            return
        logger.debug("StopNotify called")
        self.notifying = False

        # stop the periodic update
        if self.notify_timer:
            GLib.source_remove(self.notify_timer)
            self.notify_timer = None

    def notify_client(self):
        if not self.notifying:
            logger.debug("Not notifying, skipping notify_client")
            return

        result_bytes = json.dumps(self.value).encode('utf-8')
        logger.debug("Notifying values changed")
        self.PropertiesChanged(
        GATT_CHRC_IFACE,
        {"Value": [dbus.Byte(b) for b in result_bytes]},
        []
        )

    def check_battery(self):
        if not self.notifying:
            return False  # stops the GLib timer

        self.value = return_battery_power()
        logger.debug("Battery updated: %s", self.value)
        self.notify_client()

        return True  # continue calling periodically


# class PowerOffCharacteristic(Characteristic):
#     uuid = ""
#     description = b"Write to power off BLE!"

#     def __init__(self, bus, index, service):
#         Characteristic.__init__(
#             self, bus, index, self.uuid, ["write"], service,
#         )
#         self.add_descriptor(CharacteristicUserDescriptionDescriptor(bus, 3, self))

#     def WriteValue(self, options):
#         # CHANGE IN FUTURE TO POWER DOWN DEVICE
        


class CharacteristicUserDescriptionDescriptor(Descriptor):
    """
    Writable CUD descriptor.
    """

    CUD_UUID = "2901"

    def __init__(
        self, bus, index, characteristic,
    ):

        self.value = array.array("B", characteristic.description)
        self.value = self.value.tolist()
        Descriptor.__init__(self, bus, index, self.CUD_UUID, ["read"], characteristic)

    def ReadValue(self, options):
        return self.value

    def WriteValue(self, value, options):
        if not self.writable:
            raise NotPermittedException()
        self.value = value


class rpiAdvertisement(Advertisement):
    def __init__(self, bus, index):
        Advertisement.__init__(self, bus, index, "peripheral")

        self.add_service_uuid(rpiService.rpi_SVC_UUID)

        self.add_local_name("Company17_Rpi5")
        self.include_tx_power = True


def register_ad_cb():
    logger.info("Advertisement registered")


def register_ad_error_cb(error):
    logger.critical("Failed to register advertisement: " + str(error))
    mainloop.quit()

AGENT_PATH = "/com/bluez/agent"


def main():
    global mainloop
    
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SystemBus()

    adapter = find_adapter(bus)
    if not adapter:
        logger.critical('GattManager1 interface not found')
        return

    adapter_obj = bus.get_object(BLUEZ_SERVICE_NAME, adapter)

    adapter_props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")

    # powered property on the controller to on
    adapter_props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(1))

    # Get manager objs
    service_manager = dbus.Interface(adapter_obj, GATT_MANAGER_IFACE)
    ad_manager = dbus.Interface(adapter_obj, LE_ADVERTISING_MANAGER_IFACE)

    advertisement = rpiAdvertisement(bus, 0)
    obj = bus.get_object(BLUEZ_SERVICE_NAME, "/org/bluez")

    agent = Agent(bus, AGENT_PATH)

    app = Application(bus)
    app.add_service(rpiService(bus, 2))

    mainloop = MainLoop()

    agent_manager = dbus.Interface(obj, "org.bluez.AgentManager1")
    agent_manager.RegisterAgent(AGENT_PATH, "NoInputNoOutput")

    ad_manager.RegisterAdvertisement(
        advertisement.get_path(),
        {},
        reply_handler=register_ad_cb,
        error_handler=register_ad_error_cb,
    )

    logger.info("Registering GATT application...")

    service_manager.RegisterApplication(
        app.get_path(),
        {},
        reply_handler=register_app_cb,
        error_handler=register_app_error_cb,
    )

    agent_manager.RequestDefaultAgent(AGENT_PATH)

    try:
        mainloop.run()
    except KeyboardInterrupt:
        print("Interrupted. Cleaning up...")
    finally:
        ad_manager.UnregisterAdvertisement(advertisement)
        dbus.service.Object.remove_from_connection(advertisement)
        agent.Release()
        mainloop.quit()


if __name__ == "__main__":
    main()

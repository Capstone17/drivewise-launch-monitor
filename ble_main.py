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

from video_ball_detector import process_video
from metrics.ruleBasedSystem import rule_based_system

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
        self.add_characteristic(SwingAnalysisCharacteristic(bus, 0, self))
        self.add_characteristic(GenerateFeedbackCharacteristic(bus, 1, self))


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

        try:
            # Run script
            subprocess.run(
                [
                    "./embedded/GScrop_improved_flip.sh",
                    "400",
                    "144",
                    "387",
                    "3000",
                    "400",
                ],
                check=True,
            )
            logger.info("processing video now")
            # Process video
            result = process_video(
                "tst.mp4",
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
        self.notify_client()

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

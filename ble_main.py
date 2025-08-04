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

from video_ball_detector import process_video

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
        self.value = {'face angle': None, 'swing path': None, 'attack angle': None, 'side angle': None}
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
            # ./GScrop_improved_flip.sh 816 144 387 2000 2300
            subprocess.run(
                [
                    "./GScrop_improved_flip.sh",
                    "816",
                    "144",
                    "387",
                    "3000",
                    "500",
                ],
                check=True,
            )
            video_path = os.path.expanduser("~/Documents/test/tst.mp4")
            try:
                process_video(
                    video_path,
                    "ball_coords.json",
                    "sticker_coords.json",
                    "stationary_sticker.json",
                )
            except Exception as e:
                logger.error(f"Video processing failed: {e}")
            # Worst case scenario
            logger.debug("Updated value after script")

            if self.notifying:
                self.notify_client()

        except Exception as e:
            self.value = {'face angle': 100, 'swing path': 500, 'attack angle': 100, 'side angle': 900}
            logger.error(f"Failed to process write: {e}")
        
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
    uuid = "047ba9de-bf0f-4aa6-be91-3856aec11c01"
    description = b"Generate feedback based on swing metrics!"

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index, self.uuid, ["read"], service,
        )
        self.value = "No swing detected! Please try again."
        self.add_descriptor(CharacteristicUserDescriptionDescriptor(bus, 1, self))

    def ReadValue(self, options):
        # take text from json file that has feedback
        self.value = "Straight draw: A gentle rightward path with a square face is causing a draw. If your shot is landing too far left of the target, try slightly weakening your grip or evening out your path."
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
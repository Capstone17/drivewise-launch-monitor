#include <sdbus-c++/sdbus-c++.h> 
#include <iostream>
#include <vector>

using namespace std;
using namespace sdbus;

// Represents a single GATT characteristic to be exported on D-BUS
class TestCharacteristic {
public:
    // Constructor: binds characteristic to a D-Bus object path and system bus connection
    TestCharacteristic(IConnection& conn, const std::string& path)
        : obj_(createObject(conn, path)) {

        // Sets up D-Bus read method to allow remote devices to read from this characteristic over BLE
        obj_->registerMethod("ReadValue")
            .onInterface("org.bluez.GattCharacteristic1")
            .implementedAs([this](const std::map<std::string, Variant>& options) {
                std::vector<uint8_t> value{42}; // Sample value
                return value;
            });

        // Sets up D-Bus write method to allow remote devices to write to this characteristic over BLE
        obj_->registerMethod("WriteValue")
            .onInterface("org.bluez.GattCharacteristic1")
            .implementedAs([this](const std::vector<uint8_t>& value, const std::map<std::string, Variant>& options) {
                std::cout << "Write received: ";
                for (auto byte : value) std::cout << (int)byte << " ";
                std::cout << std::endl;
            });

        // Characteristic Identifier (should eventually be variable)
        obj_->registerProperty("UUID")
            .onInterface("org.bluez.GattCharacteristic1")
            .withGetter([]() { return std::string("12345678-1234-5678-1234-56789abcdef0"); });

        // Path to parent service (fixed)
        obj_->registerProperty("Service")
            .onInterface("org.bluez.GattCharacteristic1")
            .withGetter([]() { return ObjectPath("/com/example/service0"); });

        // Represents the supported actions (read & write)
        obj_->registerProperty("Flags")
            .onInterface("org.bluez.GattCharacteristic1")
            .withGetter([]() { return std::vector<std::string>{"read", "write"}; });

        obj_->finishRegistration();
    }

private:
    std::unique_ptr<IObject> obj_;
};

// Represents a GATT service, grouping one or several GATT characteristics
class TestService {
public:
    // Constructor: Binds GATT service to D-Bus path and connection
    TestService(IConnection& conn, const std::string& path)
        : obj_(createObject(conn, path)) {

        // Unique service identifier
        obj_->registerProperty("UUID")
            .onInterface("org.bluez.GattService1")
            .withGetter([]() { return std::string("12345678-1234-5678-1234-56789abcdef1"); });

        // Marks service as primary
        obj_->registerProperty("Primary")
            .onInterface("org.bluez.GattService1")
            .withGetter([]() { return true; });

        // Exposes associated characteristics as a list
        obj_->registerProperty("Characteristics")
            .onInterface("org.freedesktop.DBus.Properties")
            .withGetter([]() { return std::vector<ObjectPath>{ObjectPath("/com/example/service0/char0")}; });

        obj_->finishRegistration();
    }

private:
    std::unique_ptr<IObject> obj_;
};

int main() {
    // Create D-Bus connection to (either the session or system) bus and requests a well-known name on it.
    auto connection = sdbus::createSystemBusConnection("com.example.ble");

    // Create Bluez D-Bus object.

    // auto connection = createSystemBusConnection();
    // connection->requestName("com.example.ble");

    TestService service(*connection, "/com/example/service0");
    TestCharacteristic characteristic(*connection, "/com/example/service0/char0");

    std::cout << "BLE Peripheral running. Press Ctrl+C to quit..." << std::endl;
    connection->enterEventLoop();

    return 0;
}

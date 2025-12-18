# DriveWise Launch Monitor – Bluetooth API

## Overview

DriveWise exposes a Bluetooth Low Energy (BLE) GATT service that allows a mobile app to control swing analysis, retrieve device information, and run calibration routines. The mobile app connects to the advertised peripheral and interacts with five characteristics, each handling a specific function. All multi-byte payloads are transmitted as UTF-8 encoded JSON.

## Service Definition

- **Service Name**: DriveWise RPi Service
- **Service UUID**: `96f0284d-8895-4c08-baaf-402a2f7e8c5b`
- **Advertised Local Name**: `group17rpi`

## Characteristics

The service exposes five characteristics, each with specific read/write/notify properties:

| Characteristic Name | UUID | Properties | Direction | Purpose |
|---|---|---|---|---|
| Swing Analysis | `d9c146d3-df83-49ec-801d-70494060d6d8` | write, notify | App → RPi, RPi → App | Start swing analysis and receive golf metrics and feedback |
| Find IP | `2c75511d-11b8-407d-b275-a295ef2c199f` | read | RPi → App | Retrieve the current IP address of the device |
| Calibration | `778c5d1a-315f-4baf-a23b-6429b84835e3` | write, notify | App → RPi, RPi → App | Trigger exposure and crop calibration routines |
| Battery Monitor | `a834f0f7-89cc-453b-8be4-2905d27344bf` | read, notify | RPi → App | Get battery status (read) or subscribe to periodic updates (notify) |
| Cancel Swing | `8f1a5ff0-399b-4afe-9cb4-280c8310e388` | write | App → RPi | Cancel an ongoing swing analysis or camera capture |

---

## Detailed Characteristic Specifications

### Swing Analysis Characteristic

**UUID**: `d9c146d3-df83-49ec-801d-70494060d6d8`  
**Properties**: write, notify

#### Description

This characteristic controls the main swing analysis workflow. Writing to it initiates a two-stage capture-and-analysis process:
1. Low-rate ball detection to identify when a swing begins
2. High-rate video capture and analysis to extract metrics

Upon completion, two separate notifications are sent: one with golf metrics and one with feedback.

#### Write Behavior

- **Trigger**: Write any non-empty value to this characteristic
- **Behavior**: 
  - If capture is not already running, starts the `swingAnalysisLoop` in a background thread
  - If capture is already running, logs a message and returns (no duplicate captures)
- **Write Payload**: Arbitrary data (e.g., a single byte `0x01` or empty array). The payload content is not interpreted.

#### Notify Behavior

Two notifications are sent upon swing analysis completion or failure:

##### Notification 1: Metrics (JSON)

Sent as UTF-8 encoded JSON bytes:

```json
{
  "type": "metrics",
  "face angle": 2.45,
  "swing path": -1.23,
  "attack angle": 18.5,
  "side angle": 0.87,
  "group": "mid-iron"
}
```

| Field | Type | Description |
|---|---|---|
| `type` | string | Always `"metrics"` |
| `face angle` | float (degrees) | Club face angle relative to target |
| `swing path` | float (degrees) | Direction of swing relative to target line |
| `attack angle` | float (degrees) | Angle of club head approaching the ball (vertical) |
| `side angle` | float (degrees) | Club head angle relative to ball position (horizontal) |
| `group` | string | Inferred club type (e.g., `"mid-iron"`, `"driver"`) |

If analysis fails, `metrics` is `null` and `feedback` contains an error message.

##### Notification 2: Feedback (JSON)

Sent immediately after the metrics notification, also as UTF-8 encoded JSON bytes:

```json
"Straight: Now that is a strike!"
```

The feedback is a JSON-encoded string (or can be a more complex object if enhanced in future versions). It provides coaching or diagnostic feedback based on the detected swing characteristics.

#### Error Handling

If any step of the analysis fails (e.g., ball detection, video processing, inference), the metrics notification will contain `null` for metrics and the feedback notification will contain an error description, such as:

```json
"Could not capture the ball"
```

Or:

```json
"Swing analysis failed! Please try again."
```

---

### Find IP Characteristic

**UUID**: `2c75511d-11b8-407d-b275-a295ef2c199f`  
**Properties**: read

#### Description

Returns the current IP address(es) of the launch monitor host device.

#### Read Behavior

- **Return Value**: UTF-8 encoded JSON string of the output from `hostname -I`
- **Example Response**:
  ```json
  "192.168.1.123 "
  ```

This is useful for the mobile app to establish a connection or communicate with the host over a local network.

#### Error Handling

If the IP lookup fails, an error string is returned as JSON.

---

### Calibration Characteristic

**UUID**: `778c5d1a-315f-4baf-a23b-6429b84835e3`  
**Properties**: write, notify

#### Description

Triggers calibration routines to optimize camera exposure and crop settings for the current lighting and setup conditions.

#### Write Behavior

- **Trigger**: Write any non-empty value to this characteristic
- **Behavior**: 
  - Runs exposure calibration (adjusts camera shutter speed for optimal image brightness)
  - Runs crop calibration (adjusts the region of interest based on fiducial marker position)
- **Write Payload**: Arbitrary data. The payload is not interpreted.

#### Notify Behavior

On success, sends:

```json
"success"
```

On failure, sends an error message as a JSON-encoded string:

```json
"Calibration function failed: [error details]"
```

Or:

```json
"GS crop script failed: [error details]"
```

#### Typical Usage

1. App writes any value to trigger calibration
2. Device logs calibration progress and runs scripts
3. Device sends either `"success"` or an error string via notify
4. App updates UI to reflect calibration status

---

### Battery Monitor Characteristic

**UUID**: `a834f0f7-89cc-453b-8be4-2905d27344bf`  
**Properties**: read, notify

#### Description

**Note**: Battery monitoring is not fully implemented at this time. This characteristic is reserved for future use. Reads and notifications may return placeholder or incomplete data.

#### Read Behavior (if implemented)

- Returns battery status from `return_battery_power()` as UTF-8 encoded JSON
- Expected format: percentage string or structured object, e.g., `"85%"` or `{"battery_percent": 85}`

#### Notify Behavior (if implemented)

- When notifications are enabled, sends periodic battery status updates approximately every 60 seconds
- Stops sending when notifications are disabled

---

### Cancel Swing Characteristic

**UUID**: `8f1a5ff0-399b-4afe-9cb4-280c8310e388`  
**Properties**: write

#### Description

Cancels any ongoing swing analysis, camera capture, or calibration routine.

#### Write Behavior

- **Trigger**: Write any non-empty value to this characteristic
- **Behavior**: 
  - Clears the `camera_event` flag, which halts:
    - Low-rate ball detection loop
    - High-rate video capture
    - Analysis processing
  - Logs cancellation message
- **Write Payload**: Arbitrary data. The payload is not interpreted.
- **Response**: No response or notification is sent. Check the Swing Analysis characteristic's metrics for `null` to confirm cancellation.

#### Typical Usage

1. App writes any value to cancel
2. Device stops any active capture/analysis
3. Device logs the cancellation
4. On next read of Swing Analysis, metrics will be `null` or analysis will have been halted

---

## Connection Flow Example

### Typical swing analysis flow:

1. **App connects** to the BLE peripheral `group17rpi`
2. **App discovers** the service UUID `96f0284d-8895-4c08-baaf-402a2f7e8c5b`
3. **App enables notifications** on the Swing Analysis characteristic
4. **App writes** any value (e.g., `0x01`) to the Swing Analysis characteristic to start
5. **Device begins** low-rate ball detection and logs progress
6. **Device detects** ball and transitions to high-rate capture
7. **Device processes** video and runs rule-based analysis
8. **Device sends notification 1** with metrics (golf angles, club type)
9. **Device sends notification 2** with feedback (coaching message)
10. **App disables notifications** and displays results to the user

### Calibration flow:

1. **App enables notifications** on the Calibration characteristic
2. **App writes** any value to trigger calibration
3. **Device runs** exposure and crop calibration scripts
4. **Device sends notification** with `"success"` or error message
5. **App updates UI** based on success/failure

---

## Notes

- All JSON payloads are UTF-8 encoded bytes transmitted over BLE
- Characteristics with `notify` property require the app to explicitly enable notifications (via GATT client configuration descriptor) to receive unsolicited updates
- The Swing Analysis characteristic is blocking: only one swing analysis can run at a time. Subsequent write requests while analysis is in progress are ignored
- All characteristics are part of a single service; the app must discover and interact with all five to use the full DriveWise functionality

---

## Future Enhancements

- Battery monitoring (currently reserved, not fully implemented)
- Streaming video preview characteristic
- Configuration characteristic to set camera parameters remotely
- Statistics characteristic to retrieve historical shot data
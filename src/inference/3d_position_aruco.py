import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation vector to roll, pitch and yaw in degrees.

    The rotation returned by OpenCV assumes the marker Z axis points out of the
    board. When the marker is parallel to the camera this results in a 180° roll
    offset. Multiplying by a flip matrix aligns both coordinate systems so that a
    front facing marker yields zero rotation."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    rotation_matrix = flip @ rotation_matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    return tuple(np.degrees([roll, pitch, yaw]))

# ArUco marker ayarları
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
marker_length = 1.75  # Marker uzunluğu (cm)

# Kamera kalibrasyon parametreleri (örnek değerler)
camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros(5)

# Kamerayı başlat
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# 3D grafik hazırlığı
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X ekseni (cm)")
ax.set_ylabel("Y ekseni (cm)")
ax.set_zlabel("Z ekseni (cm)")
ax.set_title("Marker merkezine göre kamera konumu")

# Radar chart for roll, pitch and yaw
fig_radar, ax_radar = plt.subplots(subplot_kw={"polar": True})
angles = np.linspace(0, 2 * np.pi, 4)
radar_line, = ax_radar.plot([], [], marker="o")
ax_radar.set_ylim(-180, 180)
ax_radar.set_xticks(angles[:3])
ax_radar.set_xticklabels(["Roll", "Pitch", "Yaw"])

# Kamera konumlarını takip etmek için bir liste
camera_positions = []
marker_detected = False  # Markerın tespit edilip edilmediğini takip eder

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı!")
        break

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ArUco markerlarını algıla
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    if ids is not None:
        # Marker tespit edildi, pozisyonları eklemeye devam et
        marker_detected = True

        # Markerları çiz
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Her marker için poz tahmini yap
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Marker merkezine göre kameranın pozisyonunu hesapla
            tvec = tvecs[i][0]
            rvec = rvecs[i][0]

            # X, Y, Z değerlerini al
            x, y, z = tvec

            # Marker merkezine göre kameranın uzaklığını hesapla
            distance = np.linalg.norm(tvec)

            # Orientation in roll, pitch, yaw (degrees)
            roll, pitch, yaw = rvec_to_euler(rvec)

            # Kameranın konumunu kaydet
            camera_positions.append((x, y, z))

            # Marker merkezini bul
            marker_center = (
                int(corners[i][0][:, 0].mean()),
                int(corners[i][0][:, 1].mean()),
            )

            # X, Y, Z konum bilgilerini görüntünün üzerine yazdır
            cv2.putText(
                frame,
                f"X: {x:.2f} cm",
                (marker_center[0], marker_center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Y: {y:.2f} cm",
                (marker_center[0], marker_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Z: {z:.2f} cm",
                (marker_center[0], marker_center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Distance: {distance:.2f} cm",
                (marker_center[0], marker_center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Orientation values
            cv2.putText(
                frame,
                f"Roll: {roll:.2f}",
                (marker_center[0], marker_center[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Pitch: {pitch:.2f}",
                (marker_center[0], marker_center[1] + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Yaw: {yaw:.2f}",
                (marker_center[0], marker_center[1] + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            # Update radar plot
            radar_line.set_data(angles, [roll, pitch, yaw, roll])

            # 3D grafikte önceki tüm konumlarla bir çizgi oluştur
            ax.cla()  # Grafiği temizle
            ax.set_xlabel("X ekseni (cm)")
            ax.set_ylabel("Y ekseni (cm)")
            ax.set_zlabel("Z ekseni (cm)")
            ax.set_title("Marker merkezine göre kamera konumu")

            # Markerı merkezde sabitle
            ax.scatter(0, 0, 0, color="r", s=100)  # Marker merkezi
            ax.text(
                0, 0, 0, f"Marker ID: {ids[i][0]}", color="black", fontsize=10
            )  # Marker ID'si

            # Kamera pozisyonlarını çiz
            if len(camera_positions) > 1:
                positions = np.array(camera_positions)
                ax.plot(
                    positions[:, 0], positions[:, 1], positions[:, 2], color="b"
                )  # Mavi çizgi
                ax.scatter(
                    positions[-1, 0], positions[-1, 1], positions[-1, 2], color="g"
                )  # Son pozisyon yeşil nokta

            plt.draw()
            plt.pause(0.001)

    else:
        # Marker tespit edilmediğinde pozisyonları sıfırla
        if marker_detected:
            camera_positions.clear()
            marker_detected = False

    # Görüntüyü göster
    cv2.imshow("Kamera Görüntüsü", frame)

    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
plt.show()
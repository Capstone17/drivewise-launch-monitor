import subprocess
from pathlib import Path
from .histogram import analyze_exposure_in_folder


def calibrate_exposure():

    # Define the exposure sample photo path
    exposure_samples_path = Path("~/Documents/webcamGolf/embedded/exposure_samples/").expanduser()
    exposure_samples_path_as_str = str(exposure_samples_path) + "/"

    # Define the commands as a list of lists
    # commands = [
    #     ['echo', 'Starting command series...'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '50_exposure.jpg', '--shutter', '50', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '100_exposure.jpg', '--shutter', '100', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '150_exposure.jpg', '--shutter', '150', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '200_exposure.jpg', '--shutter', '200', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '250_exposure.jpg', '--shutter', '250', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '300_exposure.jpg', '--shutter', '300', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '350_exposure.jpg', '--shutter', '350', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '400_exposure.jpg', '--shutter', '400', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '450_exposure.jpg', '--shutter', '450', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '500_exposure.jpg', '--shutter', '500', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '550_exposure.jpg', '--shutter', '550', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '600_exposure.jpg', '--shutter', '600', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '650_exposure.jpg', '--shutter', '650', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '700_exposure.jpg', '--shutter', '700', '--gain', '1', '--immediate'],
    #     ['rpicam-still', '-o', exposure_samples_path_as_str + '750_exposure.jpg', '--shutter', '750', '--gain', '1', '--immediate']
    # ]

    # Define the commands as a list of lists
    commands = [
        ['echo', 'Starting command series...'],
        ['echo', 'Capturing exposures...'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '50_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '50', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '100_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '100', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '150_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '150', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '200_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '200', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '250_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '250', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '300_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '300', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '350_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '350', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '400_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '400', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '450_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '450', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '500_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '500', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '550_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '550', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '600_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '600', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '650_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '650', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '700_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '700', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '750_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '750', '--frames', '1'],
        ['rpicam-vid', '-o', exposure_samples_path_as_str + '2500_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '2500', '--frames', '1'],
        ['echo', 'Extracting frames...'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '50_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '50_exposure.jpg'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '100_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '100_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '150_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '150_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '200_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '200_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '250_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '250_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '300_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '300_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '350_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '350_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '400_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '400_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '450_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '450_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '500_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '500_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '550_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '550_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '600_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '600_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '650_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '650_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '700_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '700_exposure.jpg', '-y'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '750_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '750_exposure.jpg', '-y']
        , ['ffmpeg', '-y', '-loglevel', 'error', '-i', exposure_samples_path_as_str + '2500_exposure.mp4', '-frames:v', '1', '-update', '1', exposure_samples_path_as_str + '2500_exposure.jpg', '-y']           
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            print(f"Command: {' '.join(cmd)}")
            print(f"Stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {' '.join(cmd)}")
            print(f"Exit Code: {e.returncode}")
            print(f"Stderr:\n{e.stderr}")
            break
        except FileNotFoundError:
            print(f"Error: Command not found - {' '.join(cmd)}")
            break

    print("Command series finished.")

    folder_path = Path("~/Documents/webcamGolf/embedded/exposure_samples/").expanduser()
    exposures, brightness, best_exposure = analyze_exposure_in_folder(folder_path, visualize=True)
    rounded_best_exposure = round(best_exposure)

    print("Exposures:", exposures)
    print("Brightness per image:", brightness)
    print(f"Estimated Ideal Exposure: {best_exposure:.2f}")

    print(f"Rounded best exposure: {rounded_best_exposure}")

    return str(rounded_best_exposure)


# if __name__ == "__main__":
#     calibrate_exposure()

import subprocess
from pathlib import Path
from histogram import analyze_exposure_in_folder

# Define the commands as a list of lists, where each inner list represents a command and its arguments.
# For example, ['ls', '-l'] represents the command 'ls -l'.
commands = [
    ['echo', 'Starting command series...'],
    ['rpicam-still', '-o', 'exposure_samples/50_exposure.jpg', '--shutter', '50' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/100_exposure.jpg', '--shutter', '100' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/150_exposure.jpg', '--shutter', '150' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/200_exposure.jpg', '--shutter', '200' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/250_exposure.jpg', '--shutter', '250' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/300_exposure.jpg', '--shutter', '300' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/350_exposure.jpg', '--shutter', '350' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/400_exposure.jpg', '--shutter', '400' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/450_exposure.jpg', '--shutter', '450' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/500_exposure.jpg', '--shutter', '500' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/550_exposure.jpg', '--shutter', '550' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/600_exposure.jpg', '--shutter', '600' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/650_exposure.jpg', '--shutter', '650' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/700_exposure.jpg', '--shutter', '700' ' --gain', '1', '--immediate'],
    ['rpicam-still', '-o', 'exposure_samples/750_exposure.jpg', '--shutter', '750' ' --gain', '1', '--immediate']]

for cmd in commands:
    try:
        # Run each command.
        # `capture_output=True` captures stdout and stderr.
        # `text=True` decodes the output as text.
        # `check=True` raises a CalledProcessError if the command returns a non-zero exit code.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"Command: {' '.join(cmd)}")
        print(f"Stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Stderr:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Exit Code: {e.returncode}")
        print(f"Stderr:\n{e.stderr}")
        break  # Stop execution if a command fails
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


# Generate a .txt file where the ideal exposure can be stored
#   This eliminates the need for a global variable inside the BLE script
output_file_name = "ideal_exposure.txt"
with open(output_file_name, 'w') as file:
    file.write(str(rounded_best_exposure))

print(f"File '{output_file_name}' created and written successfully.")

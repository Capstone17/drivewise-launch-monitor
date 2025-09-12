import subprocess

# Define the commands as a list of lists, where each inner list represents a command and its arguments.
# For example, ['ls', '-l'] represents the command 'ls -l'.
commands = [
    ['echo', 'Starting command series...'],
    ['rpicam-still', '-o', '100_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '200_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '300_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '400_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '500_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '600_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '700_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '800_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '900_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate'],
    ['rpicam-still', '-o', '1000_exposure.jpg', '--shutter', '500' '--gain', '1', '--awbgains', '1,1', '--immediate']
    ] 

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

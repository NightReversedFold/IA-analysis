import requests
import subprocess
import os
import platform

url = 'http://127.0.0.1:8000/video/'
input_video_path = 'arre.mp4' # Make sure this file exists in the same directory or provide the correct path
output_video_path = 'processed_video.mp4'

try:
    with open(input_video_path, 'rb') as f:
        file_to_upload = {'file': (os.path.basename(input_video_path), f, 'video/mp4')}
        print(f"Sending request to {url} with file {input_video_path}...")
        resp = requests.post(url=url, files=file_to_upload, stream=True) 

    if resp.status_code == 200:
        print("Video processed successfully. Saving to disk...")
        with open(output_video_path, 'wb') as out_file:
            for chunk in resp.iter_content(chunk_size=8192):
                out_file.write(chunk)
        print(f"Processed video saved as {output_video_path}")

        # Open the video file
        try:
            current_os = platform.system()
            if current_os == 'Darwin':  # macOS
                subprocess.run(['open', output_video_path], check=True)
            elif current_os == 'Windows':
                # os.startfile is Windows specific, but less safe. subprocess.run is preferred.
                subprocess.run(['cmd', '/c', 'start', output_video_path], check=True, shell=True)
            elif current_os == 'Linux':
                subprocess.run(['xdg-open', output_video_path], check=True)
            else:
                print(f"Unsupported OS: {current_os}. Please open {output_video_path} manually.")
        except FileNotFoundError: # Handle if 'open', 'cmd', or 'xdg-open' is not found
            print(f"Could not find a command to open the video. Please open {output_video_path} manually.")
        except subprocess.CalledProcessError as e:
            print(f"Command to open video failed: {e}. Please open {output_video_path} manually.")
        except Exception as e:
            print(f"Failed to open video file automatically: {e}. Please open {output_video_path} manually.")

    else:
        print(f"Error processing video. Status code: {resp.status_code}")
        try:
            error_data = resp.json()
            print("Error details:", error_data)
        except requests.exceptions.JSONDecodeError:
            print("Error details (raw):", resp.text)

except FileNotFoundError:
    print(f"Error: Input video file '{input_video_path}' not found. Make sure it's in the same directory as the script or provide the full path.")
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: Could not connect to the server at {url}. Ensure the FastAPI server is running.")
    print(f"Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
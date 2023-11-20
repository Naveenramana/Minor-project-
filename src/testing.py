import pyaudio

p = pyaudio.PyAudio()

num_devices = p.get_device_count()

print("Available audio devices:\n")
for i in range(num_devices):
    device_info = p.get_device_info_by_index(i)
    print(device_info, "\n")


p.terminate()

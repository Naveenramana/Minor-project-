from SpeechToText import SpeechToText
from PreprocessDataset import *
from multiprocessing import Queue
from multiprocessing import Process
import socket
import logging
from threading import Thread


def process_stream(private_key_file_path, output_queue, CHANNELS, RATE, device_index):
    stt = SpeechToText(private_key_file_path, CHANNELS, RATE, device_index)

    for transcript, non_modified_transcript in stt.recognize_speech_stream():
        # If transcript is a list of strings, join them into a single string
        if isinstance(transcript, list) and all(isinstance(s, str) for s in transcript):
            transcript = ' '.join(transcript)
        # Check if transcript is a non-empty string
        if isinstance(transcript, str) and len(transcript) > 0:
            transcript_words = word_tokenize(transcript)

        stemmed_words = stem_strings(transcript_words, 'en')
        output_queue.put(stemmed_words)


def run_process_stream(private_key_file_path, output_queue, CHANNELS, RATE, device_index):
    try:
        process_stream(private_key_file_path, output_queue,
                       CHANNELS, RATE, device_index)
    except KeyboardInterrupt:
        print(f"Stopping process for device {device_index}")
    except Exception as e:
        print(f"Exception in process for device {device_index}: {e}")

# Function to handle client connections and send data


def client_handler(connection, microphone_queue, loopback_queue, logger):
    while True:
        # TODO: FIX the issue where data from queues need to leave at the same time.
        microphone_data = microphone_queue.get()
        loopback_data = loopback_queue.get()

        combined_data = {
            "Attacker_Helper": loopback_data,
            "Victim": microphone_data
        }

        print(
            f"Sending data to PySpark Structured Streaming application: {combined_data}")

        logger.info(
            "SpeechToText_Streaming_Server - sending data: %s", combined_data)

        message = str(combined_data) + '\n'
        connection.sendall(message.encode())

# Function to start the socket server


def start_server(host, port, microphone_queue, loopback_queue):

    # Setting up the logging
    log_format = "%(asctime)s - %(name)s - %(message)s"
    logging.basicConfig(filename="G:\\Dissertation_Project\\Logs\\performance_logs.log",
                        level=logging.INFO, format=log_format)
    logger = logging.getLogger("Dissertation_Project")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()

    print(f"Starting server on {host}:{port}")

    while True:
        conn, addr = server.accept()
        print(
            f"----------------------Accepted connection from {addr}----------------------")
        Thread(target=client_handler, args=(
            conn, microphone_queue, loopback_queue, logger)).start()


if __name__ == "__main__":

    private_key_file_path = 'Environment\speech-to-text.json'

    microphone_queue = Queue()
    loopback_queue = Queue()

    process_microphone = Process(target=run_process_stream, args=(
        private_key_file_path, microphone_queue, 1, 44100, 1,))

    process_loopback = Process(target=run_process_stream, args=(
        private_key_file_path, loopback_queue, 1, 44100, 3,))

    process_microphone.start()
    process_loopback.start()

    start_server('localhost', 9999, microphone_queue, loopback_queue)

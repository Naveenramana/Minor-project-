import os
import pyaudio
import io
import json
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK data if needed
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')


class SpeechToText:

    def __init__(self, private_key_file_path) -> None:
        self.transcript_list = []
        self.non_modified_transcript_list = []
        api_key_json = self._read_json_file(private_key_file_path)

        credentials = service_account.Credentials.from_service_account_info(
            json.loads(api_key_json))
        self.client = speech.SpeechClient(credentials=credentials)

    # Private Function
    def _read_json_file(self, file_path):
        with open(file=file_path, mode='r') as file:
            return file.read()

    def get_transcript_list(self):
        return self.transcript_list

    # Private Function
    def _preprocess_transcripts(self, transcript):

        def process_string(s):
            s = re.sub(r'\d', 'X', s)
            s = ''.join(c for c in s if c.isalpha() or c == 'X' or c == ' ')
            s = s.lower()
            s = s.lstrip()
            return s

        # Removal of stop words in the word list after the tokenization
        def remove_stopwords(words):
            stop_words = set(stopwords.words("english"))
            filtered_words = [word for word in words if word not in stop_words]
            return filtered_words

        def stem_words(words):
            stemmer = PorterStemmer()

            # Stem each word
            stemmed_words = [stemmer.stem(word) for word in words]
            return stemmed_words

        processed_string = process_string(transcript)

        # Tokenize the string into words
        tokenized_words = word_tokenize(processed_string)

        # Remove stopwords
        removed_stopwords = remove_stopwords(words=tokenized_words)

        # Stem the words
        stemmed_words = stem_words(words=removed_stopwords)

        final_transcript = ' '.join(stemmed_words)

        return final_transcript

    def recognize_speech_stream(self):

        # Setting up PyAudio for capturing microphone input signals
        FORMAT = pyaudio.paInt16  # Good choice for trade-off between precision and memory usage

        CHANNELS = 1
        RATE = 16000
        CHUNK = int(RATE/10)

        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("Start speaking... (Press Ctrl+C to stop)")

        def stream_generator():
            while not self.stop_flag:
                data = stream.read(CHUNK)
                yield speech.StreamingRecognizeRequest(audio_content=data)

        self.stop_flag = False
        requests = stream_generator()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US")

        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True)

        try:
            responses = self.client.streaming_recognize(
                streaming_config, requests)

            for response in responses:
                for result in response.results:
                    if result.is_final and result.alternatives:
                        transcript = result.alternatives[0].transcript

                        self.non_modified_transcript_list.append(transcript)

                        transcript = self._preprocess_transcripts(
                            transcript=transcript)

                        self.transcript_list.append(transcript)

                        print("\nTranscript_List:\t", self.transcript_list)
                        print("Non modified Transcript_List:\t",
                              self.non_modified_transcript_list)

        except KeyboardInterrupt:
            self.stop_flag = True
            print("\nStopping...")

        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":

    private_key_file_path = 'Environment\speech-to-text.json'

    speechToText = SpeechToText(private_key_file_path)
    transcript_list = speechToText.recognize_speech_stream()

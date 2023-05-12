from SpeechToText import SpeechToText
import threading
from queue import Queue
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from nltk.tokenize import word_tokenize
import nltk


def process_stream(private_key_file_path, processed_transcripts):
    
    stt = SpeechToText(private_key_file_path)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    for transcript, non_modified_transcript in stt.recognize_speech_stream():
        
        # If transcript is a list of strings, join them into a single string
        if isinstance(transcript, list) and all(isinstance(s, str) for s in transcript):
            transcript = ' '.join(transcript)

        # Check if transcript is a non-empty string
        if isinstance(transcript, str) and len(transcript) > 0:
            transcript_words = word_tokenize(transcript)
        
        # print(transcript_words)

        # Fit and transform the tokenized words into one-hot encoded vectors
        encoder.fit(np.array(transcript_words).reshape(-1, 1))
        transcript_one_hot = encoder.transform(np.array(transcript_words).reshape(-1, 1))

        # Add the one-hot encoded vectors to the queue
        processed_transcripts.put((transcript_one_hot, non_modified_transcript))
    
    

if __name__ == "__main__":
    private_key_file_path = 'Environment\speech-to-text.json'
    
    processed_transcripts = Queue()
    
    # Start the SpeechToText thread
    stt_thread = threading.Thread(target=process_stream, args=(private_key_file_path, processed_transcripts))
    stt_thread.start()

    try:
        while stt_thread.is_alive():
            # Check if there is a preprocessed transcript in the queue
            if not processed_transcripts.empty():
                (transcript, non_modified_transcript) = processed_transcripts.get()
                print("Received preprocessed transcript (one-hot encoded):\n", transcript)
                print("Received non_modified_transcript:", non_modified_transcript)
                

    except KeyboardInterrupt:
        print("Stopping...")
        
        
    stt_thread.join()
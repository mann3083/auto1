let conversationIndex = 0;
const conversation = [
    { question: "What is your name?", answer: null },
    { question: "Please provide your policy number.", answer: null },
    { question: "What is your date of birth?", answer: null },
    // Add more questions as needed
];

document.getElementById("recordButton").onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    let chunks = [];

    mediaRecorder.ondataavailable = (e) => {
        chunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', blob, 'audio.webm');

        try {
            const response = await fetch('/transcribe/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            document.getElementById("transcription").textContent = data.transcription;

            if (data.transcription.trim().length > 0) {
                conversation[conversationIndex].answer = data.transcription;
                conversationIndex++;
                if (conversationIndex < conversation.length) {
                    displayQuestion();
                } else {
                    alert("End of conversation. Thank you!");
                }
            } else {
                alert("Please speak again, your response was not clear.");
            }

        } catch (error) {
            console.error('Error during transcription:', error);
            document.getElementById("transcription").textContent = 'Transcription failed.';
        }
    };

    mediaRecorder.start();
    document.getElementById("recordButton").textContent = "Recording...";
    document.getElementById("recordButton").disabled = true;

    setTimeout(() => {
        mediaRecorder.stop();
        document.getElementById("recordButton").textContent = "Start Speaking";
        document.getElementById("recordButton").disabled = false;
    }, 3000); // Stop recording after 3 seconds of silence
};

document.getElementById("textToSpeechForm").addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = document.getElementById("text").value;

    try {
        const response = await fetch('/tts/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        const audioElement = `<audio controls><source src="${data.audio_url}" type="audio/mp3">Your browser does not support the audio element.</audio>`;
        document.getElementById("audioPlayer").innerHTML = audioElement;

    } catch (error) {
        console.error('Error during text-to-speech conversion:', error);
    }
});

function displayQuestion() {
    document.getElementById("prompt").textContent = conversation[conversationIndex].question;
    document.getElementById("transcription").textContent = '';
}

const questions = [
    "What is your name?",
    "Please provide your policy number.",
    //"What is your date of birth?",
    // Add more questions as needed
];

let currentQuestionIndex = 0;

async function fetchTTS(text) {
   
    //console.log(typeof(text))
    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch('/tts/', {
        method: 'POST',
        body: formData
        //headers: {
        //    'Content-Type': 'application/json'
        //},
        //body: JSON.stringify({ text: text })
    });

    const responseText = await response.text();
    console.log('Response Text:', responseText);  // Log the response text for debugging

    if (!response.ok) {
        throw new Error('Failed to fetch TTS audio');
    }

    try {
        
        const data = JSON.parse(responseText);  // Parse the response text as JSON
        return data.audio_url;

    } catch (error) {
        console.error('Error parsing JSON:', error, 'Response Text:', responseText);
        throw new Error('Failed to parse JSON response');
    }
}


async function playNextQuestion() {
    if (currentQuestionIndex >= questions.length) {
        document.getElementById('question').textContent = "Processing details..."
        return;
    };

    const questionText = questions[currentQuestionIndex];
    document.getElementById('question').textContent = questionText;

    try {
        const audioUrl = await fetchTTS(questionText);
        const audioElement = `<audio controls autoplay>
                                <source id="audioSource" src="${audioUrl}" type="audio/mp3">
                                Your browser does not support the audio element.
                              </audio>`;
        document.getElementById('audioPlayer').innerHTML = audioElement;
    } catch (error) {
        console.error('Error fetching TTS audio:', error);
        //alert('Failed to play the question audio.');
    }

    currentQuestionIndex++;

    // LOGIC TO RECORD USER VOICE AFTER THE QUESTION IS ASKED
    startRecording()
    setTimeout(() => {
        stopRecording();
    }, 15000); // 15 seconds timeout

    //Reccursively call the next function with new quetsion text/
    setTimeout(playNextQuestion, 3000); // Wait 5 seconds before playing the next question
}

document.getElementById('startButton').onclick = () => {
    playNextQuestion();
    document.getElementById('startButton').disabled = true;
};


async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('file', audioBlob, 'audio.webm');

            try {
                const response = await fetch('/transcribe/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();

                document.getElementById("transcription").innerText = data.transcription;

                // After transcription, proceed to the next question
                currentQuestionIndex++;
                setTimeout(playNextQuestion, 5000); // Wait 5 seconds before playing the next question

            } catch (error) {
                console.error('Error during transcription:', error);
                document.getElementById("transcription").innerText = 'Transcription failed.';
            }
        };

        // Start recording immediately
        mediaRecorder.start();
        audioChunks = []; // Clear any existing audio chunks
        timeoutID = setTimeout(() => {
            stopRecording();
        }, 15000); // Automatically stop recording after 15 seconds

    } catch (error) {
        console.error('Error starting recording:', error);
    }
}

function stopRecording() {
    if (mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        clearTimeout(timeoutID); // Clear the timeout
        audioChunks = []; // Reset audio chunks for the next recording
    }
}
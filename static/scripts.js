const questions = [
    "What is your name?",
    "Please provide your policy number.",
    "What is your date of birth?",//document.getElementById("dob").value='2023-01-01'
    // Add more questions as needed
];

let currentQuestionIndex = 0;
let mediaRecorder;
let audioChunks = [];

async function fetchTTS(text) {
    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch('/tts/', {
        method: 'POST',
        body: formData
    });

    const responseText = await response.text();

    if (!response.ok) {
        throw new Error('Failed to fetch TTS audio');
    }

    try {
        const data = JSON.parse(responseText);
        return data.audio_url;
    } catch (error) {
        console.error('Error parsing JSON:', error, 'Response Text:', responseText);
        throw new Error('Failed to parse JSON response');
    }
}

async function playNextQuestion() {
    if (currentQuestionIndex >= questions.length) {
        //document.getElementById('question').textContent = "Processing details...";
        return;
    }

    const questionText = questions[currentQuestionIndex];
    document.getElementById('question').textContent = questionText;


    try {
        const audioUrl = await fetchTTS(questionText);
        const audioElement = `<audio controls autoplay>
                                <source id="audioSource" src="${audioUrl}" type="audio/mp3">
                                Your browser does not support the audio element.
                              </audio>`;
        document.getElementById('audioPlayer').innerHTML = audioElement;

        const audio = document.getElementById('audioSource').parentElement;
        audio.onended = startRecording; // Start recording after TTS ends

    } catch (error) {
        console.error('Error fetching TTS audio:', error);
        // Optionally handle the error, e.g., display a message to the user
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

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
                
                //NAME
                let policyName = document.getElementById('name');
                if (policyName === '') {
                    policyName.value=data.transcription
                }

                //POLICY NUMBER
                let policyNumberInput = document.getElementById('policyNumber');
                if (policyNumberInput === '') {
                    policyNumberInput.value=data.transcription
                }

                //DOB

                let policyDOB = document.getElementById('dob');
                if (policyDOB === '') {
                    policyDOB.value=data.transcription
                }
                

                currentQuestionIndex++;
                setTimeout(playNextQuestion, 5000); // Wait 5 seconds before playing the next question

            } catch (error) {
                console.error('Error during transcription:', error);
                document.getElementById("transcription").innerText = 'Transcription failed.';
            }
        };

        mediaRecorder.start();

        // Automatically stop recording after 15 seconds or if silence is detected
        setTimeout(() => {
            if (mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        }, 5000);

    } catch (error) {
        console.error('Error starting recording:', error);
    }
}

document.getElementById('startButton').onclick = () => {
    playNextQuestion();
    document.getElementById('startButton').disabled = true;
};

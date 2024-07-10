let mediaRecorder;
let audioChunks = [];

document.getElementById("recordButton").onclick = async () => {
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
            
        } catch (error) {
            console.error('Error during transcription:', error);
            document.getElementById("transcription").innerText = 'Transcription failed.';
        }
    };
    
    mediaRecorder.start();
    document.getElementById("recordButton").disabled = true;
    document.getElementById("stopButton").disabled = false;
};

document.getElementById("stopButton").onclick = () => {
    mediaRecorder.stop();
    document.getElementById("recordButton").disabled = false;
    document.getElementById("stopButton").disabled = true;
    audioChunks = [];
};

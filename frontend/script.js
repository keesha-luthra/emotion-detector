// ================= MULTIMODAL STATE =================
let multimodalState = {
    textEmotion: null,
    speechEmotion: null,
    faceEmotion: null,
    text: "",
    speech: "",
    image: null
};

// ================= TEXT =================
async function analyze() {
    let text = document.getElementById("textInput").value.trim();

    if (!text) {
        alert("Please enter text");
        return;
    }

    multimodalState.text = text;

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    });

    const result = await response.json();

    multimodalState.textEmotion = result.emotion;

    document.getElementById("resultText").innerHTML =
        `Emotion: ${result.emotion} <br> Confidence: ${result.confidence}`;

    updateSummary();
}


// ================= SPEECH =================
function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        alert("Use Chrome");
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";

    recognition.start();

    recognition.onresult = async function(event) {
        let text = event.results[0][0].transcript;

        multimodalState.speech = text;

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const result = await response.json();

        multimodalState.speechEmotion = result.emotion;

        document.getElementById("resultSpeech").innerHTML =
            `Speech: ${text}<br>Emotion: ${result.emotion}`;
        
        updateSummary();
    };
}


// ================= FACE =================
let faceStarted = false;

async function startFace() {
    const video = document.getElementById("video");
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

setInterval(() => {
    if (location.hash === "#face" && !faceStarted) {
        faceStarted = true;
        startFace();
    }
}, 500);


// ================= FACE DETECTION =================
async function detectEmotion() {
    const video = document.getElementById("video");

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");

    multimodalState.image = imageData;

    const response = await fetch("http://127.0.0.1:5000/predict-face", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: imageData })
    });

    const result = await response.json();

    multimodalState.faceEmotion = result.emotion;

    document.getElementById("resultFace").innerHTML =
        `Emotion: ${result.emotion} <br> Confidence: ${result.confidence}`;

    updateSummary();
}


// ================= 🔥 FUSION =================
async function runFusion() {
    const response = await fetch("http://127.0.0.1:5000/fuse", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: multimodalState.text,
            speech: multimodalState.speech,
            image: multimodalState.image
        })
    });

    const result = await response.json();

    document.getElementById("finalEmotion").innerText = result.emotion;
    document.getElementById("finalConfidence").innerText = result.confidence;

    let breakdown = "";
    for (let key in result.details) {
        breakdown += `${key}: ${result.details[key].toFixed(2)}\n`;
    }

    document.getElementById("fusionDetails").innerText = breakdown;
}


// ================= SUMMARY =================
function updateSummary() {

    if (multimodalState.textEmotion) {
        document.getElementById("textResult").innerText = multimodalState.textEmotion;
        document.getElementById("textBar").style.width = "70%";
    }

    if (multimodalState.speechEmotion) {
        document.getElementById("speechResult").innerText = multimodalState.speechEmotion;
        document.getElementById("speechBar").style.width = "70%";
    }

    if (multimodalState.faceEmotion) {
        document.getElementById("faceResultSummary").innerText = multimodalState.faceEmotion;
        document.getElementById("faceBar").style.width = "70%";
    }

    // 🔥 run fusion AFTER individual results
    runFusion();
}
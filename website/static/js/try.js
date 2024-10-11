// for image upload and prediction

document.getElementById('uploadImg').onsubmit = async function(event) {
    console.log("hi");
    event.preventDefault();
    document.getElementById('viewimgResult').innerHTML = 'Processing image...';
    document.getElementById('viewimgResult').style="display:block";
    document.getElementById('imageResult').innerHTML = `<div><p>Loading......</p></div>`;
    const formData = new FormData();
    const imageFile = document.getElementById('imageInput').files[0];
    formData.append('image', imageFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.message && result.message === "No faces detected") {
            // If no faces are detected, show the message
            document.getElementById('imageResult').innerHTML = `<div><p>No faces detected in the image.</p></div>`;
            document.getElementById('viewimgResult').innerHTML = 'View Result';
            document.getElementById('viewimgResult').onclick = function() {
                alert("No faces detected in the image.");
            }
        } else if (result.emotions && result.image) {
            // If faces are detected, show the emotion and image
            let emotionOutput = '<p>Predicted Emotions:</p><ul>';

            result.emotions.forEach(function(item) {
                emotionOutput += `<li>${item.emotion}</li>`;
            });

            emotionOutput += '</ul>';
            document.getElementById('imageResult').innerHTML = `
                <div>
                    ${emotionOutput}
                    <img src="data:image/png;base64,${result.image}" alt="Predicted Image" style="max-width: 100%; height: auto;"/>
                </div>`;
            document.getElementById('viewimgResult').href="#imageResult";
            document.getElementById('viewimgResult').innerHTML = 'View Result';
        } else {
            // Fallback if something unexpected happens
            document.getElementById('imageResult').innerHTML = `<div><p>Unable to detect emotion. Please try again.</p></div>`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('imageResult').innerHTML = `<p>There was an error processing your request.</p>`;
    }
};

// for video upload and prediction

document.getElementById('uploadVideo').onsubmit = async function(event) {
    event.preventDefault();
    document.getElementById('viewvideoResult').innerHTML = 'Processing video...';
    document.getElementById('viewvideoResult').style="display:block";
    document.getElementById('videoResult').innerHTML = `<div><p>Loading.....</p></div>`;
    const formData = new FormData();
    const videoFile = document.getElementById('videoInput').files[0];
    formData.append('video', videoFile);

    try {
        const response = await fetch('/predict-video', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            document.getElementById('viewvideoResult').href="#videoResult";
            document.getElementById('viewvideoResult').innerHTML = 'View Result';
            const videoBlob = await response.blob();
            const videoURL = URL.createObjectURL(videoBlob);

            document.getElementById('videoResult').innerHTML = `
                <video controls width="600">
                    <source src="${videoURL}" type="video/mp4">
                </video>`;
        } else {
            const result = await response.json();
            document.getElementById('videoResult').innerHTML = `<p>Error: ${result.error}</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('videoResult').innerHTML = `<p>There was an error processing your request.</p>`;
    }
};

// for webcam prediction

const video = document.getElementById('web-cam');
function startVideoStream() {
    video.src = '/video_feed';
    videoStream = true; 
    video.style.display = 'block';
}
function stopVideoStream() {
    videoStream = false;  // Stop fetching frames
    video.src = '';
    video.style.display = 'none';  
}


document.addEventListener('keydown', function(event) {
    if (event.key === 'q' || event.key === 'Q') {
        stopVideoStream();
    }
});
document.getElementById('webcam').addEventListener('click',function(){
    startVideoStream();
    location.href="#web-cam";
});

document.getElementById('webcaminfo').addEventListener('click',function(){
    alert("It will initally take some time to load the webcam, please wait for a few seconds.\nIf asked for permission, please allow the browser to access the webcam.\nPress 'Q' to stop the webcam.");    
});

document.getElementById('videoinfo').addEventListener('click',function(){
    alert("If the video is greater than 20 seconds or 2MB, it will take more time to process. For better results, please upload smaller videos.");
});
// for image upload and prediction

document.getElementById('uploadImg').onsubmit = async function(event) {
    console.log("hi");
    event.preventDefault();
    document.getElementById('viewimgResult').innerHTML = 'Processing image...';
    document.getElementById('viewimgResult').style="display:block";

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
            document.getElementById('imageResult').innerHTML = `
                <div>
                    <p>Predicted Emotion: ${result.emotions[0].emotion}</p>
                    <img src="data:image/png;base64,${result.image}" alt="Predicted Image" style="max-width: 100%; height: auto;"/>
                </div>`;
            document.getElementById('viewimgResult').href="#resultcont";
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

    const formData = new FormData();
    const videoFile = document.getElementById('videoInput').files[0];
    formData.append('video', videoFile);

    try {
        const response = await fetch('/predict-video', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            document.getElementById('viewvideoResult').href="#resultcont";
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
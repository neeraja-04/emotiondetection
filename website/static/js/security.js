const video = document.getElementById('web-cam');
function startVideoStream() {
    video.src = '/haf/video_feed';
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
document.getElementById('webcam-view').addEventListener('click',function(){
    startVideoStream();
    location.href="#web-cam";
});

document.getElementById('webcaminfo').addEventListener('click',function(){
    alert("It will initally take some time to load the webcam, please wait for a few seconds.\nIf asked for permission, please allow the browser to access the webcam.\nPress 'Q' to stop the webcam.");    
});

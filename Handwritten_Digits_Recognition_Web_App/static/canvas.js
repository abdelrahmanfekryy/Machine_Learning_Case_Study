const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");
const BrushCap = document.querySelector('#brushcap');
const TextResponse =  document.getElementById("textresponse");
const BrushSize = document.getElementById("brushsize")
const BrushColor = document.getElementById("brushcolor")
const BrushSizeText = document.getElementById("brushsizetext")
const SubmitButton = document.getElementById('submitbutton')
const ClearButton = document.getElementById('clearbutton')

let painting = false;

canvas.height = 400;
canvas.width = 400;

ctx.lineWidth = 30;
ctx.strokeStyle = 'red';
ctx.lineCap = 'round';


window.addEventListener("load" , (e) => {
    
    function StartPosition(e) {
        ctx.beginPath();
        painting = true;
        Draw(e);
    }

    function EndPosition() {
        painting = false;
    }

    function Draw(e) {
        if(!painting) return;
        ctx.lineTo(e.clientX - canvas.offsetLeft,e.clientY - canvas.offsetTop + document.documentElement.scrollTop)
        ctx.stroke();

        //for smoothness
        ctx.beginPath();
        ctx.lineTo(e.clientX - canvas.offsetLeft,e.clientY - canvas.offsetTop + document.documentElement.scrollTop);
        ctx.closePath();
    }


    function ClearCanvas(){ 
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        TextResponse.innerHTML = 'Please Sketch A Digit'
    }
    
    
    function SendData(e,path='/book'){   
        var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        fetch(`${window.origin}${path}`, {
            method: "POST",
            credentials: "include",
            body: JSON.stringify(imageData),
            cache: "no-cache",
            headers: new Headers({
            "content-type": "application/json"
            })
        }).then(function (response) {
            if (response.status !== 200) {
              console.log(`Looks like there was a problem. Status code: ${response.status}`);
              return;
            }
            response.json().then(function (data) {
                TextResponse.innerHTML = `The Predicted Digit is ${data}`
            });
          }).catch(function (error) {
            console.log("Fetch error: " + error);
          });
    
        };
    
    function UpdateSize(){
        BrushSizeText.innerHTML = BrushSize.value;
        ctx.lineWidth = BrushSize.value;
    }
    
    function UpdateColor(){
        ctx.strokeStyle = BrushColor.value;
    }

    function onMouseMove(e) {
        BrushCap.setAttribute("style" , `top: ${e.clientY + document.documentElement.scrollTop - BrushSize.value/2}px;  left: ${e.clientX - BrushSize.value/2}px; background-color: ${BrushColor.value}; width: ${BrushSize.value}px; height: ${BrushSize.value}px`);
    }

    canvas.addEventListener('mousedown',StartPosition);
    document.addEventListener('mouseup', EndPosition);
    canvas.addEventListener('mousemove', Draw);
    canvas.addEventListener('mousemove', onMouseMove);
    BrushColor.addEventListener('input',UpdateColor);
    BrushSize.addEventListener('input',UpdateSize);
    SubmitButton.addEventListener('click',SendData);
    ClearButton.addEventListener('click',ClearCanvas);
});




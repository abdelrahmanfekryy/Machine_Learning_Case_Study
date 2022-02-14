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
let mymodel;

canvas.height = 400;
canvas.width = 400;

ctx.lineWidth = 30;
ctx.strokeStyle = 'red';
ctx.lineCap = 'round';

class L2 {

    static className = 'L2';

    constructor(config) {
       return tf.regularizers.l1l2(config)
    }
}
tf.serialization.registerClass(L2);



window.addEventListener("load" , (e) => {

    async function loadMobilenet() {
    const model = await tf.loadLayersModel("https://outer-projects.s3.amazonaws.com/trained_model/model.json");
    return model
    }
    mymodel = loadMobilenet();

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

    function ProcessData(e){
        var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        let values = tf.tensor3d(imageData.data, [canvas.height, canvas.width, 4]).resizeBilinear([32,32]).slice([0,0,0], [32,32,1]).div(tf.scalar(255)).expandDims(0);
        mymodel.then(function (res) {
            TextResponse.innerHTML = `The Predicted Digit is ${res.predict(values).argMax(-1).dataSync()[0]}`
        })
    }
    
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
    SubmitButton.addEventListener('click',ProcessData);
    ClearButton.addEventListener('click',ClearCanvas);
    
});




// Загрузка модели TensorFlow.js
let model;
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("Модель загружена!");
}

// Инициализация canvas
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

canvas.addEventListener('mousedown', () => isDrawing = true);
canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    recognizeDigit();
});
canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Очистка canvas
document.getElementById('clearButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = '?';
});

// Распознавание цифры
async function recognizeDigit() {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const tensor = preprocessImage(imgData);

    const prediction = model.predict(tensor);
    const predictedClass = prediction.argMax(1).dataSync()[0];
    document.getElementById('prediction').textContent = predictedClass;
}

// Предобработка изображения
function preprocessImage(imgData) {
    const grayscale = [];
    for (let i = 0; i < imgData.data.length; i += 4) {
        grayscale.push(255 - imgData.data[i]); // Инверсия цвета
    }

    const resized = tf.tensor(grayscale, [280, 280]).resizeBilinear([28, 28]);
    const normalized = resized.div(255.0).reshape([1, 28, 28, 1]);
    return normalized;
}

// Загрузка модели при запуске
loadModel();
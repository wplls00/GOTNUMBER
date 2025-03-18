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

    // Создаем тензор размером [280, 280] и добавляем канал (1)
    let tensor = tf.tensor(grayscale, [280, 280], 'float32').expandDims(-1);

    // Масштабируем до [28, 28]
    tensor = tf.image.resizeBilinear(tensor, [28, 28]);

    // Нормализуем значения пикселей (0-1)
    tensor = tensor.div(255.0);

    // Добавляем батч (размерность [1, 28, 28, 1])
    return tensor.expandDims(0);
}

// Загрузка модели при запуске
loadModel();

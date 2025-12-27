
class SnakeEnv {
    constructor(n = 5) {
        this.n = n;
        this.reset();
    }

    reset() {
        const mid = Math.floor(this.n / 2);
        this.head = [mid, mid];
        this.snake = [mid * this.n + mid, mid * this.n + (mid - 1)];
        this.dirIdx = 1;
        this.directions = [[-1, 0], [0, 1], [1, 0], [0, -1]];

        this.food = null;
        this.running = true;
        this.movimientos = 0;
        this.frutas = 0;
        this.prevHeadIdx = mid * this.n + (mid - 1);
        this.placeFood();

        return this.datos();
    }

    placeFood() {
        if (this.snake.length >= this.n * this.n) {
            this.food = null;
            return;
        }
        let idx;
        do {
            idx = Math.floor(Math.random() * this.n * this.n);
        } while (this.snake.includes(idx));
        this.food = idx;
    }

    paso(actionIdx) {
        if (!this.running) return { done: true, recompensa: 0, state: this.datos() };

        this.movimientos++;
        this.prevHeadIdx = this.snake[0];

        if (actionIdx === 1) {
            this.dirIdx = (this.dirIdx + 1) % 4;
        } else if (actionIdx === 2) {
            this.dirIdx = (this.dirIdx + 3) % 4;
        }

        const [dr, dc] = this.directions[this.dirIdx];
        const oldDist = this._getDist(this.snake[0]);

        const newR = Math.floor(this.snake[0] / this.n) + dr;
        const newC = (this.snake[0] % this.n) + dc;
        const newIdx = newR * this.n + newC;

        const isDead = this._checkCollision(newR, newC, newIdx);

        if (isDead) {
            this.running = false;
            const recompensa = this._calculateReward(false, true, oldDist, 0, newIdx);
            return { done: true, recompensa, frutas: this.frutas, state: this.datos(), steps: this.movimientos };
        }

        this.snake.unshift(newIdx);
        let ate = false;
        if (newIdx === this.food) {
            ate = true;
            this.frutas++;
            this.placeFood();
        } else {
            this.snake.pop();
        }

        const newDist = this._getDist(newIdx);
        const recompensa = this._calculateReward(ate, false, oldDist, newDist, newIdx);

        if (this.movimientos > 100 * this.snake.length) {
            this.running = false;
            return { done: true, recompensa: -1.1, frutas: this.frutas, state: this.datos(), steps: this.movimientos };
        }

        return { done: false, recompensa, frutas: this.frutas, state: this.datos(), ate, steps: this.movimientos };
    }

    _checkCollision(r, c, idx) {
        if (r < 0 || r >= this.n || c < 0 || c >= this.n) return true;
        const tailIdx = this.snake[this.snake.length - 1];
        if (this.snake.includes(idx) && !(idx === tailIdx && idx !== this.food)) return true;
        return false;
    }

    _getDist(idx) {
        if (this.food === null) return 0;
        const r = Math.floor(idx / this.n), c = idx % this.n;
        const fr = Math.floor(this.food / this.n), fc = this.food % this.n;
        return Math.sqrt(Math.pow(r - fr, 2) + Math.pow(c - fc, 2));
    }

    _calculateReward(ate, isDead, oldDist, newDist, newIdx) {
        if (isDead) return -1.0;
        if (ate) return 1.0;

        const reachable = this._getReachableCount(newIdx, this.snake);
        const totalFree = (this.n * this.n) - this.snake.length;
        const spaceRatio = totalFree > 0 ? reachable / totalFree : 0;

        let compactness = 0;
        const r = Math.floor(newIdx / this.n), c = newIdx % this.n;
        [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(([dr, dc]) => {
            const nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= this.n || nc < 0 || nc >= this.n) compactness += 0.015;
            else if (this.snake.includes(nr * this.n + nc)) compactness += 0.015;
        });

        if (spaceRatio < 0.35) return -0.25 + compactness;

        let reward = -0.025;
        reward += (newDist < oldDist) ? 0.12 : -0.15;
        return reward + compactness;
    }

    _getReachableCount(startIdx, snake) {
        const queue = [startIdx];
        const visited = new Set([startIdx]);
        const bodySet = new Set(snake.slice(0, -1));
        let count = 0;

        while (queue.length > 0) {
            const curr = queue.shift();
            count++;
            const r = Math.floor(curr / this.n), c = curr % this.n;

            [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(([dr, dc]) => {
                const nr = r + dr, nc = c + dc;
                const nIdx = nr * this.n + nc;
                if (nr >= 0 && nr < this.n && nc >= 0 && nc < this.n && !visited.has(nIdx) && !bodySet.has(nIdx)) {
                    visited.add(nIdx);
                    queue.push(nIdx);
                }
            });
        }
        return count;
    }

    datos() {
        const channels = 3;
        const state = Array.from({ length: channels }, () =>
            Array.from({ length: this.n }, () => new Array(this.n).fill(0)));

        const headIdx = this.snake[0];
        const hr = Math.floor(headIdx / this.n), hc = headIdx % this.n;

        state[0][hr][hc] = 1.0;
        if (this.prevHeadIdx !== null && this.prevHeadIdx !== headIdx) {
            const pr = Math.floor(this.prevHeadIdx / this.n), pc = this.prevHeadIdx % this.n;
            state[0][pr][pc] = 0.5;
        }

        for (let i = 1; i < this.snake.length; i++) {
            const r = Math.floor(this.snake[i] / this.n), c = this.snake[i] % this.n;
            state[1][r][c] = 1.0;
        }

        if (this.food !== null) {
            const fr = Math.floor(this.food / this.n), fc = this.food % this.n;
            state[2][fr][fc] = 1.0;
        }

        return state;
    }
}

let env;
let sharedModel = null;
let actorModel = null;
let gameInterval = null;
let isPlaying = false;
let isHumanMode = false;
let humanNextAction = 0; // 0: Recto (por defecto)

// Elementos del DOM
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreSpan = document.getElementById('scoreDisplay');
const stepSpan = document.getElementById('stepDisplay');
const btnPlay = document.getElementById('btnPlay');
const btnReset = document.getElementById('btnReset');
const sizeInput = document.getElementById('gridSizeInput');
let valor = Number(sizeInput.value);
valor= valor*valor;
maximaPaciencia = Math.floor(valor * (valor * 0.416));
const humanCheck = document.getElementById('humanMode');
const speedRange = document.getElementById('speedRange');
const URL_SHARED = "http://192.168.56.1:5550/projectos/ActorCritic/model/sharedV1.json";
const URL_ACTOR = "http://192.168.56.1:5550/projectos/ActorCritic/model/actorV1.json";

function reconstruirModeloDesdeJSON(modelData) {
    let layers = [];

    for (let i = 0; i < modelData.length; i++) {
        const layerData = modelData[i];
        const tipo = layerData["tipo"]; // "conv2d", "fc", etc.

        if (tipo === "relu") {
            layers.push(new ReLU());
        }
        else if (tipo === "softmax") {
            // Importante: El Actor usa Softmax al final
            layers.push(new Softmax());
        }
        else if (tipo === "maxPool") {
            // Asumo stride = size si no está definido, ajusta según tu constructor
            layers.push(new MaxPool2D(layerData.size, layerData.size));
        }
        else if (tipo === "conv2d") {
            let conv = new Conv2D(
                layerData.inChannels,
                layerData.outChannels,
                layerData.kernelSize,
                layerData.padding
            );
            // RESTAURAR PESOS (Vital)
            conv.kernels = layerData.kernels;
            conv.bias = layerData.bias;
            layers.push(conv);
        }
        else if (tipo === "fc" || tipo === "linear") { // A veces guardamos como 'fc' o 'linear'
            let fc = new Linear(layerData.inp, layerData.out);
            // RESTAURAR PESOS (Vital)
            fc.weight = layerData.weight;
            fc.bias = layerData.bias;
            layers.push(fc);
        }
        else if (tipo === "flat") {
            layers.push(new flat());
        }
        else if (tipo === "drop") {
            layers.push(new Dropout(layerData.p));
        }
    }

    console.log("Modelo reconstruido con", layers.length, "capas.");
    // Devolvemos la instancia real de Sequential con métodos .forward()
    return new Sequential(layers);
}
async function cargaModeloSnake(url) {
    try {
        console.log(`Descargando: ${url}...`);
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Error al descargar ${url}: ${response.status} ${response.statusText}`);
        }

        // Obtenemos el array de datos puro
        const modelData = await response.json();

        // Lo pasamos por la "fábrica" de objetos
        return reconstruirModeloDesdeJSON(modelData);

    } catch (error) {
        console.error(`Error crítico cargando modelo desde ${url}:`, error);
        alert("Error cargando la IA. Revisa la consola (F12) y asegura que el servidor esté activo.");
        return null;
    }
}
async function loadModels() {
    console.log("⏳ Descargando modelos desde el servidor local...");

    // Mostramos un mensaje en la interfaz si quieres
    const statusLabel = document.getElementById('scoreDisplay');
    statusLabel.textContent = "Cargando IA...";

    // Cargamos ambos en paralelo para ganar velocidad
    const [shared, actor] = await Promise.all([
        cargaModeloSnake(URL_SHARED),
        cargaModeloSnake(URL_ACTOR)
    ]);

    if (shared && actor) {
        sharedModel = shared;
        actorModel = actor;
        console.log("✅ IA lista para jugar.");
        statusLabel.textContent = "0";
    } else {
        alert("Error crítico: No se pudieron obtener los pesos del modelo.");
    }
}

// --- Inicialización del Juego ---
function initGame() {
    stopGame();

    // Validar tamaño impar
    let n = parseInt(sizeInput.value);
    if (n % 2 === 0) {
        n += 1;
        sizeInput.value = n;
    }

    env = new SnakeEnv(n);
    draw();
    updateStats(0, 0);
    humanNextAction = 0; // Reset acción humana
}

function gameStep() {
    if (!env.running) {
        stopGame();
        // alert(`Game Over! Frutas: ${env.frutas}, Pasos: ${env.movimientos}`);
        return;
    }

    let action = 0; // 0: Recto por defecto

    if (isHumanMode) {
        // En modo humano usamos la tecla presionada
        action = humanNextAction;
        humanNextAction = 0; // Resetear para que el siguiente frame sea recto si no se oprime nada
    } else {
        // Modo IA
        if (sharedModel && actorModel) {
            let state = env.datos(); // [3, H, W]

            // Forward
            let features = sharedModel.forward(state);
            let probs = actorModel.forward(features)[0]; // Tomamos el primer elemento del batch

            if (Number(stepSpan.textContent) > maximaPaciencia) {
                // Obtenemos los índices ordenados de mayor a menor probabilidad
                // Creamos un array de índices [0, 1, 2, 3] y los ordenamos según su valor en 'probs'
                let indicesOrdenados = probs
                    .map((prob, index) => ({ prob, index }))
                    .sort((a, b) => b.prob - a.prob);

                // Tomamos el segundo mejor (índice 1)
                action = indicesOrdenados[1].index;// osea se mata
            } else {
                // Movimiento normal: el mejor (el de mayor probabilidad)
                action = probs.indexOf(Math.max(...probs));
            }
        }
    }
    // Ejecutar paso
    let result = env.paso(action);
    draw();
    updateStats(result.frutas, result.steps);

    if (result.done) {
        stopGame();
        // Pequeño delay para pintar la muerte antes del alert
        // setTimeout(() => alert(`Game Over! Frutas: ${result.frutas}, Pasos: ${result.steps}`), 10);
    }
}

function startGame() {
    if (!env.running) initGame();
    if (isPlaying) return;

    isPlaying = true;
    btnPlay.textContent = "⏸ Pausa";

    const speed = 550 - parseInt(speedRange.value); // Invertir rango
    gameInterval = setInterval(gameStep, speed);
}

function stopGame() {
    isPlaying = false;
    btnPlay.textContent = "▶ Iniciar";
    clearInterval(gameInterval);
}

// --- Renderizado ---
function draw() {
    const n = env.n;
    const cellSize = canvas.width / n;

    // 1. Fondo (Negro suave)
    ctx.fillStyle = "#1e1e1e"; // Un poco más claro que negro absoluto para ver mejor
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 2. Dibujar Comida (Rojo Brillante)
    if (env.food !== null) {
        const fr = Math.floor(env.food / n);
        const fc = env.food % n;

        // Efecto de brillo (Glow)
        ctx.shadowBlur = 15;
        ctx.shadowColor = "#ff3333";

        ctx.fillStyle = "#ff3333";
        ctx.beginPath();
        // Dibujamos un círculo
        ctx.arc(fc * cellSize + cellSize / 2, fr * cellSize + cellSize / 2, cellSize / 2.5, 0, Math.PI * 2);
        ctx.fill();

        // Reset del brillo para lo demás
        ctx.shadowBlur = 0;
    }

    // 3. Dibujar Serpiente
    env.snake.forEach((idx, i) => {
        const r = Math.floor(idx / n);
        const c = idx % n;

        if (i === 0) {
            // --- CABEZA (Amarillo / Dorado) ---
            ctx.fillStyle = "#FFD700";
            // Opcional: Hacer la cabeza un pelín más grande
            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);

            // Ojos (Detalle estético opcional)
            ctx.fillStyle = "#000";
            const eyeSize = cellSize / 5;
            // Ojo simple al centro o ajustado (simple al centro para no complicar la rotación)
            ctx.beginPath();
            ctx.arc(c * cellSize + cellSize / 2, r * cellSize + cellSize / 2, eyeSize, 0, Math.PI * 2);
            ctx.fill();

        } else {
            // --- CUERPO (Verde Matriz) ---
            // Degradado sutil: más oscuro cerca de la cola
            // Pero un verde sólido funciona mejor para ver la forma
            ctx.fillStyle = "#32CD32"; // LimeGreen

            // Dejamos un pequeño borde (padding 1px) para ver los segmentos
            ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
        }
    });
}

function updateStats(score, steps) {
    scoreSpan.textContent = score;
    stepSpan.textContent = steps;
}

// --- Control Humano (Absoluto a Relativo) ---
// SnakeEnv usa: 0=Recto, 1=Derecha(Horario), 2=Izquierda(Anti)
document.addEventListener('keydown', (e) => {
    if (!isHumanMode || !isPlaying) return;

    // Mapa de teclas a índices de dirección absoluta
    // 0: Arriba, 1: Derecha, 2: Abajo, 3: Izquierda
    const keyMap = {
        'ArrowUp': 0, 'w': 0,
        'ArrowRight': 1, 'd': 1,
        'ArrowDown': 2, 's': 2,
        'ArrowLeft': 3, 'a': 3
    };

    if (keyMap[e.key] !== undefined) {
        const desiredDir = keyMap[e.key];
        const currentDir = env.dirIdx;

        // Calcular diferencia relativa
        // diff 0 -> Recto (0)
        // diff 1 -> Derecha (1)
        // diff 3 -> Izquierda (2)
        // diff 2 -> Hacia atrás (Inválido/Muerte inmediata o ignorar)

        let diff = (desiredDir - currentDir + 4) % 4;

        if (diff === 0) humanNextAction = 0;
        else if (diff === 1) humanNextAction = 1;
        else if (diff === 3) humanNextAction = 2;
        // Si diff es 2, intentó ir hacia atrás, mantenemos 0 (recto) o ignoramos
    }
});

// --- Event Listeners UI ---
btnPlay.addEventListener('click', () => {
    if (isPlaying) stopGame();
    else startGame();
});

btnReset.addEventListener('click', initGame);

sizeInput.addEventListener('change', initGame);

humanCheck.addEventListener('change', (e) => {
    isHumanMode = e.target.checked;
    stopGame();
    initGame();
});

speedRange.addEventListener('input', () => {
    if (isPlaying) {
        stopGame();
        startGame();
    }
});

// --- Inicio ---
// Intentar cargar modelos al inicio
loadModels();
initGame();
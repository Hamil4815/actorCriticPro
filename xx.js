
function isBatched4D(x) {
  // [N][C][H][W]
  return (
    Array.isArray(x) &&
    Array.isArray(x[0]) &&
    Array.isArray(x[0][0]) &&
    Array.isArray(x[0][0][0])
  );
}
function isBatched2D(x) {
  // [N][D]
  return Array.isArray(x) && Array.isArray(x[0]) && typeof x[0][0] === "number";
}
function ensureBatch4D(x) {
  if (isBatched4D(x)) return x;
  return [x]; // wrap single sample
}
function ensureBatch2D(x) {
  if (isBatched2D(x)) return x;
  return [x];
}
class Conv2D {
  constructor(inChannels, outChannels, kernelSize, padding = 0) {
    this.tipo = "conv2d";
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.padding = padding;

    // Output shape aproximado
    this.lastOutputShape = [this.outChannels, 0, 0]; 

    const K = kernelSize;
    // --- Inicialización He (Kaiming) ---
    const fanIn = inChannels * K * K;
    const std = Math.sqrt(2 / fanIn);

    this.kernels = new Array(outChannels);
    for (let f = 0; f < outChannels; f++) {
      this.kernels[f] = new Array(inChannels);
      for (let c = 0; c < inChannels; c++) {
        const kc = new Array(K);
        for (let ky = 0; ky < K; ky++) {
          const row = new Float32Array(K);
          for (let kx = 0; kx < K; kx++) {
            row[kx] = (Math.random() - 0.5) * 2 * std;
          }
          kc[ky] = row;
        }
        this.kernels[f][c] = kc;
      }
    }

    this.bias = new Float32Array(outChannels);
    this.gradBias = new Float32Array(outChannels);
    this.grad = new Array(outChannels);
    
    // Inicializar estructura de gradientes
    for (let f = 0; f < outChannels; f++) {
      this.grad[f] = new Array(inChannels);
      for (let c = 0; c < inChannels; c++) {
        const gkc = new Array(K);
        for (let ky = 0; ky < K; ky++) {
          gkc[ky] = new Float32Array(K);
        }
        this.grad[f][c] = gkc;
      }
    }

    this.input = null;
    this._tmpInputFlat = null;
    this._tmpDInputFlat = null;
  }
  forward(batchInput) {
    const batch = ensureBatch4D(batchInput);
    this.input = batch;

    const N = batch.length;
    const C = batch[0].length;
    const H = batch[0][0].length;
    const W = batch[0][0][0].length;
    const K = this.kernelSize;
    const P = this.padding;

    // Calcular tamaño de salida
    const outH = Math.floor(H + 2 * P - K) + 1;
    const outW = Math.floor(W + 2 * P - K) + 1;

    // --- CORRECCIÓN AQUÍ: Usamos Array estándar en lugar de Float32Array ---
    // Esto asegura que 'tensorInfo' vea la 4ta dimensión correctamente
    const output = Array.from({ length: N }, () =>
      Array.from({ length: this.outChannels }, () =>
        Array.from({ length: outH }, () => Array(outW).fill(0)) 
      )
    );

    // Buffer plano
    const inSize = C * H * W;
    if (!this._tmpInputFlat || this._tmpInputFlat.length < inSize) {
      this._tmpInputFlat = new Float32Array(inSize);
    }

    const kernels = this.kernels;
    const bias = this.bias;

    for (let n = 0; n < N; n++) {
      // 1. Aplanar entrada
      let p = 0;
      const img = batch[n];
      for (let c = 0; c < C; c++) {
        const plane = img[c];
        for (let y = 0; y < H; y++) {
          const row = plane[y];
          for (let x = 0; x < W; x++) {
            this._tmpInputFlat[p++] = row[x];
          }
        }
      }

      // 2. Convolución
      for (let f = 0; f < this.outChannels; f++) {
        const b = bias[f];
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            let sum = 0.0;
            
            for (let c = 0; c < C; c++) {
              const inBase = c * (H * W);
              const kc = kernels[f][c];

              for (let ky = 0; ky < K; ky++) {
                const iy = oy + ky - P;

                if (iy >= 0 && iy < H) {
                  const krow = kc[ky];
                  const inRowStart = inBase + iy * W;

                  for (let kx = 0; kx < K; kx++) {
                    const ix = ox + kx - P;
                    if (ix >= 0 && ix < W) {
                       sum += this._tmpInputFlat[inRowStart + ix] * krow[kx];
                    }
                  }
                }
              }
            }
            output[n][f][oy][ox] = sum + b;
          }
        }
      }
    }
    return output;
  }

  // constructor(inChannels, outChannels, kernelSize, padding = 0) {
  //   this.inChannels = inChannels;
  //   this.outChannels = outChannels;
  //   this.kernelSize = kernelSize;
  //   this.padding = padding;

  //   // Usamos Float32Array para los pesos (PLANO)
  //   // Esto es mucho más rápido para la caché del CPU
  //   const K = kernelSize;
  //   const fanIn = inChannels * K * K;
  //   const std = Math.sqrt(2 / fanIn);
    
  //   this.weights = new Float32Array(outChannels * inChannels * K * K);
  //   this.bias = new Float32Array(outChannels);
    
  //   // Gradientes
  //   this.gradWeights = new Float32Array(this.weights.length);
  //   this.gradBias = new Float32Array(outChannels);

  //   // Inicialización He
  //   for(let i=0; i<this.weights.length; i++) {
  //       this.weights[i] = (Math.random() - 0.5) * 2 * std;
  //   }
  // }

  // forward(x) {
  //   // x viene como [Batch][Channel][Height][Width] (Arrays normales)
  //   const N = x.length;
  //   const C = x[0].length;
  //   const H = x[0][0].length;
  //   const W = x[0][0][0].length;
  //   const K = this.kernelSize;
  //   const P = this.padding;
    
  //   const outH = Math.floor(H + 2 * P - K) + 1;
  //   const outW = Math.floor(W + 2 * P - K) + 1;

  //   // Guardamos input para backward
  //   this.input = x; 

  //   // Pre-calculamos índices para evitar multiplicaciones en el bucle interno
  //   const output = new Array(N);

  //   for (let n = 0; n < N; n++) {
  //     output[n] = new Array(this.outChannels);
  //     for (let f = 0; f < this.outChannels; f++) {
  //       // Usamos Float32Array para las filas de salida (más rápido de escribir)
  //       const outPlane = new Array(outH);
  //       const wOffset = f * (C * K * K); // Offset del peso para este filtro

  //       for (let oy = 0; oy < outH; oy++) {
  //         const row = new Float32Array(outW); 
  //         for (let ox = 0; ox < outW; ox++) {
            
  //           let sum = this.bias[f];

  //           // Bucle optimizado: "Flatten" mentalmente
  //           for (let c = 0; c < C; c++) {
  //             const imgPlane = x[n][c]; // Referencia directa al plano
  //             const wcOffset = wOffset + c * (K * K);

  //             for (let ky = 0; ky < K; ky++) {
  //               const iy = oy + ky - P;
  //               if (iy >= 0 && iy < H) {
  //                 const imgRow = imgPlane[iy]; // Referencia a la fila
  //                 const wkOffset = wcOffset + ky * K;
                  
  //                 for (let kx = 0; kx < K; kx++) {
  //                   const ix = ox + kx - P;
  //                   if (ix >= 0 && ix < W) {
  //                     // Acceso directo a memoria vs búsqueda de punteros
  //                     sum += imgRow[ix] * this.weights[wkOffset + kx];
  //                   }
  //                 }
  //               }
  //             }
  //           }
  //           row[ox] = sum;
  //         }
  //         outPlane[oy] = row; // Guardamos la fila TypedArray
  //       }
  //       output[n][f] = outPlane;
  //     }
  //   }
  //   return output;
  // }

  backward(dOutBatch) {
    const batch = ensureBatch4D(dOutBatch); 
    const N = batch.length;
    const C = this.input[0].length;
    const H = this.input[0][0].length;
    const W = this.input[0][0][0].length;
    const K = this.kernelSize;
    const P = this.padding;
    
    const outH = batch[0][0].length;
    const outW = batch[0][0][0].length;

    const inSize = C * H * W;
    // Verificamos buffers planos
    if (!this._tmpInputFlat || this._tmpInputFlat.length < inSize) 
      this._tmpInputFlat = new Float32Array(inSize);
    if (!this._tmpDInputFlat || this._tmpDInputFlat.length < inSize) 
      this._tmpDInputFlat = new Float32Array(inSize);

    const kernels = this.kernels;
    const grad = this.grad;
    const gradBias = this.gradBias;

    // --- CORRECCIÓN AQUÍ: Inicializar dInputs como Arrays estándar ---
    const dInputs = Array.from({ length: N }, () =>
      Array.from({ length: C }, () =>
        Array.from({ length: H }, () => Array(W).fill(0))
      )
    );

    for (let n = 0; n < N; n++) {
      // Re-aplanar entrada
      let p = 0;
      const img = this.input[n];
      for (let c = 0; c < C; c++) {
        const plane = img[c];
        for (let y = 0; y < H; y++) {
          const row = plane[y];
          for (let x = 0; x < W; x++) {
            this._tmpInputFlat[p++] = row[x];
          }
        }
      }

      this._tmpDInputFlat.fill(0);

      for (let f = 0; f < this.outChannels; f++) {
        const kcList = kernels[f];
        
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            const delta = batch[n][f][oy][ox];
            gradBias[f] += delta;

            for (let c = 0; c < C; c++) {
              const inBase = c * (H * W);
              const gkc = grad[f][c];
              const kkc = kcList[c];

              for (let ky = 0; ky < K; ky++) {
                const iy = oy + ky - P;

                if (iy >= 0 && iy < H) {
                   const inRowBase = inBase + iy * W;
                   const krow = kkc[ky];
                   const grow = gkc[ky];

                   for (let kx = 0; kx < K; kx++) {
                     const ix = ox + kx - P;

                     if (ix >= 0 && ix < W) {
                        const flatIdx = inRowBase + ix;
                        const inVal = this._tmpInputFlat[flatIdx];
                        
                        grow[kx] += inVal * delta; 
                        this._tmpDInputFlat[flatIdx] += krow[kx] * delta; 
                     }
                   }
                }
              }
            }
          }
        }
      }

      // Desempaquetar
      let q = 0;
      for (let c = 0; c < C; c++) {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            dInputs[n][c][y][x] = this._tmpDInputFlat[q++];
          }
        }
      }
    }
    return dInputs;
  }
  toJSON() {
    return {
      tipo: this.tipo,
      inChannels: this.inChannels,
      outChannels: this.outChannels,
      kernelSize: this.kernelSize,
      padding: this.padding,
      kernels: this.kernels,
      bias: this.bias,
    };
  }


}
class Linear {//mejora en la inicializacion de los pesos
  constructor(inSize, outSize) {
    this.tipo = "fc";
    this.inp = inSize;
    this.out = outSize;

    // --- CAMBIO AQUÍ: Inicialización He ---
    const std = Math.sqrt(2 / inSize);
    // --------------------------------------

    this.weight = Array.from({ length: outSize }, () =>
      Array.from({ length: inSize }, () => (Math.random() - 0.5) * 2 * std)
    );
    this.bias = Array(outSize).fill(0);

    // ... resto igual ...
    this.grad = Array.from({ length: outSize }, () => Array(inSize).fill(0));
    this.gradBias = Array(outSize).fill(0);
    this.lastInput = null;
  }
  // ... resto de métodos ...
  toJSON() {
    return {
      tipo: this.tipo,
      inp: this.inp,
      out: this.out,
      weight: this.weight,
      bias: this.bias,
    };
  }
  forward(x) {
    // console.log( 'antes',tensorInfo(x) );
    const batch = ensureBatch2D(x); // [N][inSize]
    // console.log( 'despues',tensorInfo(x) );
    this.lastInput = batch;
    const N = batch.length;
    const out = Array.from({ length: N }, () =>
      Array(this.weight.length).fill(0)
    ); // [N][outSize]
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < this.weight.length; i++) {
        let sum = this.bias[i];
        for (let j = 0; j < this.weight[i].length; j++)
          sum += this.weight[i][j] * batch[n][j];
        out[n][i] = sum;
      }
    }
    // console.log( 'despues',tensorInfo(out) );
    return out;
  }

  backward(dout) {
    const dBatch = ensureBatch2D(dout); // [N][outSize]
    const N = dBatch.length;
    const inSize = this.weight[0].length;
    const outSize = this.weight.length;

    // NOTE: do NOT reset this.grad here. It must accumulate across batch processing.

    // dx: [N][inSize]
    const dx = Array.from({ length: N }, () => Array(inSize).fill(0));

    for (let n = 0; n < N; n++) {
      for (let i = 0; i < outSize; i++) {
        const g = dBatch[n][i];
        this.gradBias[i] += g;
        for (let j = 0; j < inSize; j++) {
          this.grad[i][j] += this.lastInput[n][j] * g;
          dx[n][j] += this.weight[i][j] * g;
        }
      }
    }

    return dx;
  }
}
class flat {
  constructor() {
    this.tipo = "flat";
    this.lastShape = null;
  }
  toJSON() {
    return {
      tipo: this.tipo,
      lastShape: this.lastShape,
    };
  }
  forward(batchTensor3D) {
    const N = batchTensor3D.length;
    const C = batchTensor3D[0].length;
    const H = batchTensor3D[0][0].length;
    const W = batchTensor3D[0][0][0].length;

    this.lastShape = [C, H, W]; // <-- se guarda para el backward

    const OUT = Array.from({ length: N }, () => new Float32Array(C * H * W));
    for (let n = 0; n < N; n++) {
      let idx = 0;
      for (let c = 0; c < C; c++)
        for (let i = 0; i < H; i++)
          for (let j = 0; j < W; j++) {
            OUT[n][idx++] = batchTensor3D[n][c][i][j];
          }
    }
    // console.log( OUT.map((a) => Array.from(a)) );
    return OUT.map((a) => Array.from(a)); // convertir a array normal
  }

  backward(flatBatch) {
    // console.log( flatBatch[0] );
    if (!this.lastShape) throw new Error("Flat: lastShape vacío en backward");

    const [C, H, W] = this.lastShape;
    const N = flatBatch.length;
    const out = [];

    for (let n = 0; n < N; n++) {
      let idx = 0;
      const tensor = [];
      for (let c = 0; c < C; c++) {
        const channel = [];
        for (let i = 0; i < H; i++) {
          const row = [];
          for (let j = 0; j < W; j++) {
            row.push(flatBatch[n][idx++]);
          }
          channel.push(row);
        }
        tensor.push(channel);
      }
      out.push(tensor);
    }
    return out;
  }
}
class ReLU {
  constructor() {
    this.tipo = "relu";
    this.mask = null;
  }

  forward(x) {
    // x can be [N][D], [N][C][H][W], or single sample
    if (isBatched2D(x)) {
      const batch = x;
      this.mask = batch.map((row) => row.map((v) => v <= 0));
      return batch.map((row) => row.map((v) => Math.max(0, v)));
    }
    if (isBatched4D(x)) {
      const batch = x;
      this.mask = batch.map((sample) =>
        sample.map((ch) => ch.map((row) => row.map((v) => v <= 0)))
      );
      return batch.map((sample) =>
        sample.map((ch) => ch.map((row) => row.map((v) => Math.max(0, v))))
      );
    }
    // single-case fallback
    if (typeof x[0] === "number") {
      this.mask = x.map((v) => v <= 0);
      return x.map((v) => Math.max(0, v));
    }
    // conv single-sample
    this.mask = x.map((ch) => ch.map((row) => row.map((v) => v <= 0)));
    return x.map((ch) => ch.map((row) => row.map((v) => Math.max(0, v))));
  }
  toJSON() {
    return {
      tipo: this.tipo,
    };
  }
  backward(gradOutput) {
    if (isBatched2D(gradOutput)) {
      return gradOutput.map((row, n) =>
        row.map((g, i) => (this.mask[n][i] ? 0 : g))
      );
    }
    if (isBatched4D(gradOutput)) {
      return gradOutput.map((sample, n) =>
        sample.map((ch, c) =>
          ch.map((row, r) => row.map((g, k) => (this.mask[n][c][r][k] ? 0 : g)))
        )
      );
    }
    // single-case
    if (typeof gradOutput[0] === "number")
      return gradOutput.map((g, i) => (this.mask[i] ? 0 : g));
    return gradOutput.map((ch, c) =>
      ch.map((row, r) => row.map((g, k) => (this.mask[c][r][k] ? 0 : g)))
    );
  }
}
class Softmax {
  constructor() {
    this.tipo = "softmax";
    this.output = null; // Necesario guardarlo para el backward
  }

  toJSON() {
    return { tipo: this.tipo };
  }

  forward(x) {
    // Usamos la lógica de tu 'softmaxBatch'
    const batch = ensureBatch2D(x); // [N][C]

    // Calculamos y guardamos en this.output para usarlo en el backward
    this.output = batch.map((logits) => {
      const maxLogit = Math.max(...logits); // Estabilidad numérica
      const exps = logits.map((v) => Math.exp(v - maxLogit));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => e / sum);
    });

    return this.output;
  }

  backward(gradOutput) {
    // gradOutput es el gradiente que viene de la pérdida
    // La derivada de Softmax es compleja: S_i * (grad_i - sum(S_k * grad_k))

    const N = gradOutput.length;
    const C = gradOutput[0].length;

    // Preparamos matriz de salida
    const dInput = Array.from({ length: N }, () => new Array(C));

    for (let n = 0; n < N; n++) {
      // 1. Calcular producto punto (S . grad) para esta muestra
      let sumSGrad = 0;
      for (let c = 0; c < C; c++) {
        sumSGrad += this.output[n][c] * gradOutput[n][c];
      }

      // 2. Aplicar fórmula del gradiente
      for (let i = 0; i < C; i++) {
        const s = this.output[n][i];
        dInput[n][i] = s * (gradOutput[n][i] - sumSGrad);
      }
    }
    return dInput;
  }
}
class Sequential {
  constructor(...layers) {
    this.layers = layers.flat();
    this.training = true; // por defecto en modo training
  }

  train() {
    this.training = true;
  }

  eval() {
    this.training = false;
  }

  add(layer) {
    this.layers.push(layer);
  }

  forward(x) {
    // console.log( x );
    let out = x;
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      if (typeof layer.forward !== "function")
        throw new Error("Layer sin forward detectada");
      if (layer.forward.length >= 2) {
        out = layer.forward(out, this.training);
      } else {
        out = layer.forward(out);
      }
    }
    return out;
  }

  backward(grad) {
    let g = grad;
    // console.log( '-----------------------' );
    // console.log( this.layers );
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i];
      if (typeof layer.backward === "function") {
        g = layer.backward(g);
      }
    }
    return g;
  }

  getLayers() {
    return this.layers;
  }
  toJSON() {
    return this.layers
      .map((layer) =>
        typeof layer.toJSON === "function" ? layer.toJSON() : null
      )
      .filter((l) => l !== null); // evita Dropout u otros que no quieras guardar
  }

  save(path) {
    const fs = require("fs");
    fs.writeFileSync(path, JSON.stringify(this.toJSON()));
    console.log("Modelo guardado en:", path);
  }
}




/* global window, document, fetch, WebGL2RenderingContext */
/**
 * Van Gogh neural-preserving renderer - WebGL2 engine.
 *
 * Two render modes over the same CV asset bundle ( exported by
 * inference/export_web.py ):
 *
 *  1. Brush Strokes - GPU-instanced Hertzmann brush dabs ( strokes.bin )
 *     advected by the RAFT flow field, layered by semantic depth.
 *  2. Flow Warp     - layer-aware depth-parallax composite of the RGBA
 *     semantic layers with flow-driven sampling offsets.
 *
 * Both modes sample colours exclusively from the original painting -
 * motion is purely geometric ( "neural preserving" by construction ).
 *
 * No external dependencies; WebGL2 only, with a static fallback image.
 */
"use strict";

(() => {

  // ---------------------------------------------------------------------------
  // tiny helpers
  // ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  const state = {
    scene: null,
    scenes: [],
    mode: "brush",                 // 'brush' | 'warp'
    paused: false,
    time: 0,                       // animation seconds ( speed-scaled )
    wallClock: 0,
    brushAlpha: 0,                 // fade-in for the stroke pass
    params: {
      parallax: 1.0,
      flow: 0.9,
      stroke: 1.0,
      speed: 1.0,
      brush: 1.0,
      autoPan: 0.6,
    },
    camera: {
      pan: [0, 0], targetPan: [0, 0],      // px, auto-pan + user drag combined
      drag: [0, 0], dragStart: null,
      zoom: 1.0, targetZoom: 1.0,
      pointer: [0, 0],                     // normalized -0.5..0.5
    },
    fps: { ema: 60, last: performance.now() },
  };

  // ---------------------------------------------------------------------------
  // shader loading with #include resolution ( include-once )
  // ---------------------------------------------------------------------------
  const SHADER_DIRS = ["shaders/", "../shaders/"];

  async function fetchShader(name) {
    for (const dir of SHADER_DIRS) {
      try {
        const resp = await fetch(dir + name);
        if (!resp.ok) continue;
        return await resp.text();
      } catch (err) { /* try next dir */ }
    }
    throw new Error(`shader not found: ${name}`);
  }

  async function loadShaderSource(name, included = new Set()) {
    if (included.has(name)) return "";          // include-once semantics
    included.add(name);
    let src = await fetchShader(name);
    const re = /^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$/gm;
    const jobs = [];
    let m;
    while ((m = re.exec(src)) !== null) jobs.push(m[1]);
    for (const dep of jobs) {
      const depSrc = await loadShaderSource(dep, included);
      src = src.replace(new RegExp(`^[ \\t]*#include[ \\t]+"${dep}"[ \\t]*$`, "m"), depSrc);
    }
    return src;
  }

  // ---------------------------------------------------------------------------
  // GL plumbing
  // ---------------------------------------------------------------------------
  const canvas = $("app");
  /** @type {WebGL2RenderingContext} */
  let gl = null;

  function initGL() {
    gl = canvas.getContext("webgl2", {
      antialias: true,
      alpha: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    return !!gl;
  }

  function compile(type, source, label) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      throw new Error(`shader compile failed [${label}]:\n${log}\n--- source head ---\n${source.slice(0, 400)}`);
    }
    return sh;
  }

  function link(vsSource, fsSource, label) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource, label + ".vert"));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource, label + ".frag"));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`program link failed [${label}]: ${gl.getProgramInfoLog(prog)}`);
    }
    return prog;
  }

  function uniforms(prog, names) {
    const out = {};
    for (const n of names) out[n] = gl.getUniformLocation(prog, n);
    return out;
  }

  // ---------------------------------------------------------------------------
  // textures
  // ---------------------------------------------------------------------------
  function makeTexture(image, { linear = true } = {}) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`image failed: ${url}`));
      img.src = url;
    });
  }

  async function loadTexture(url, opts) {
    return makeTexture(await loadImage(url), opts);
  }

  // ---------------------------------------------------------------------------
  // scene loading
  // ---------------------------------------------------------------------------
  const assets = {
    original: null,
    layers: [],          // [{tex, depth}]
    depth: null,
    flow: null,
    strokes: null,       // Float32Array ( N * 12 )
    strokeCount: 0,
    manifest: null,
  };

  let quadVAO = null, strokeVAO = null, cornerBuf = null, instanceBuf = null;
  let progWarp = null, progFlat = null, progBrush = null;
  let uWarp = null, uFlat = null, uBrush = null;

  async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch failed: ${url} (${resp.status})`);
    return resp.json();
  }

  function disposeScene() {
    for (const t of [assets.original, assets.depth, assets.flow, ...assets.layers.map((l) => l.tex)]) {
      if (t) gl.deleteTexture(t);
    }
    if (instanceBuf) gl.deleteBuffer(instanceBuf);
    if (strokeVAO) gl.deleteVertexArray(strokeVAO);
    instanceBuf = null; strokeVAO = null;
    assets.layers = []; assets.strokes = null; assets.strokeCount = 0;
  }

  async function loadScene(sceneId) {
    showOverlay("Loading scene…", `fetching assets for “${sceneId}”`);
    disposeScene();

    const manifest = await fetchJSON(`assets/${sceneId}/manifest.json`);
    assets.manifest = manifest;

    const base = `assets/${sceneId}/`;
    assets.original = await loadTexture(base + manifest.original);
    assets.depth = await loadTexture(base + manifest.depth.file);
    assets.flow = await loadTexture(base + manifest.flow.file);

    const maxLayers = 6;
    for (let i = 0; i < Math.min(manifest.layers.length, maxLayers); i++) {
      const layer = manifest.layers[i];
      const tex = await loadTexture(base + layer.file);
      assets.layers.push({ tex, depth: layer.depth ?? 0.5, name: layer.name });
    }

    // ---- stroke instances ----
    const segCount = manifest.strokes.count;
    if (segCount > 0) {
      const resp = await fetch(base + manifest.strokes.bin);
      if (!resp.ok) throw new Error(`strokes.bin failed (${resp.status})`);
      const buf = await resp.arrayBuffer();
      const floats = new Float32Array(buf);
      const stride = manifest.strokes.strideFloats || 12;
      const records = floats.length / stride;
      // defensive stable sort by layer index ( paint order for blending )
      // record layout: x,y,dirx,diry | len,radius,opacity,layer | r,g,b,phase
      const LAYER_IDX = 7;
      const order = new Array(records).fill(0).map((_, i) => i);
      order.sort((a, b) => floats[a * stride + LAYER_IDX] - floats[b * stride + LAYER_IDX]);
      const sorted = new Float32Array(records * stride);
      for (let r = 0; r < records; r++) {
        sorted.set(floats.subarray(order[r] * stride, (order[r] + 1) * stride), r * stride);
      }
      assets.strokes = sorted;
      assets.strokeCount = records;
    }

    buildStrokeVAO();
    buildQuadVAO();

    state.camera.pan = [0, 0];
    state.camera.targetPan = [0, 0];
    state.camera.zoom = state.camera.targetZoom = 1.0;
    state.brushAlpha = 0;

    $("strokeCount").textContent = `${assets.strokeCount.toLocaleString()} strokes`;
    const models = manifest.models || {};
    $("engineInfo").textContent =
      `${models.sam || "SAM"} · ${models.depth || "depth"} · ${models.flow || "RAFT"} ` +
      `· ${manifest.width}×${manifest.height}px`;
    hideOverlay();
  }

  // ---------------------------------------------------------------------------
  // geometry
  // ---------------------------------------------------------------------------
  function buildQuadVAO() {
    if (quadVAO) { gl.bindVertexArray(quadVAO); return; }
    // unit quad: aPos (0..1) is scaled to painting px by the vertex shader
    const quad = new Float32Array([
      0, 0, 0, 0,
      1, 0, 1, 0,
      0, 1, 0, 1,
      1, 1, 1, 1,
    ]);
    quadVAO = gl.createVertexArray();
    gl.bindVertexArray(quadVAO);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);   // aPos ( 0..1 )
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);   // aUv
  }

  function buildStrokeVAO() {
    if (!assets.strokes) return;
    const corners = new Float32Array([
      -1, -1, 1, -1, -1, 1, 1, 1,
    ]);
    cornerBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);

    instanceBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, assets.strokes, gl.STATIC_DRAW);

    strokeVAO = gl.createVertexArray();
    gl.bindVertexArray(strokeVAO);

    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const stride = 12 * 4;
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
    for (let i = 1; i <= 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 4, gl.FLOAT, false, stride, (i - 1) * 16);
      gl.vertexAttribDivisor(i, 1);
    }
  }

  // ---------------------------------------------------------------------------
  // camera / view
  // ---------------------------------------------------------------------------
  const view = { scale: [1, 1], center: [0, 0] };

  function updateView() {
    const cw = Math.max(1, canvas.width);
    const ch = Math.max(1, canvas.height);
    const W = assets.manifest.width;
    const H = assets.manifest.height;
    const fit = Math.min(cw / W, ch / H) * 0.9 * state.camera.zoom;
    view.scale = [2 * fit / cw, 2 * fit / ch];
    view.center = [W / 2 - state.camera.pan[0], H / 2 - state.camera.pan[1]];
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
  }

  // ---------------------------------------------------------------------------
  // draw passes
  // ---------------------------------------------------------------------------
  function drawFlatQuad(tex, dim, time) {
    gl.useProgram(progFlat);
    gl.bindVertexArray(quadVAO);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(uFlat.uTex, 0);
    gl.uniform1f(uFlat.uDim, dim);
    gl.uniform1f(uFlat.uTime, time);
    gl.uniform2f(uFlat.uCameraCenter, view.center[0], view.center[1]);
    gl.uniform2f(uFlat.uScale, view.scale[0], view.scale[1]);
    gl.uniform2f(uFlat.uImageSize, assets.manifest.width, assets.manifest.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function drawWarp(time) {
    gl.useProgram(progWarp);
    gl.bindVertexArray(quadVAO);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, assets.original);
    gl.uniform1i(uWarp.uOriginal, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, assets.depth);
    gl.uniform1i(uWarp.uDepth, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, assets.flow);
    gl.uniform1i(uWarp.uFlow, 2);

    const count = Math.min(assets.layers.length, 6);
    for (let i = 0; i < 6; i++) {
      const loc = gl.getUniformLocation(progWarp, `uLayer${i}`);
      const dloc = gl.getUniformLocation(progWarp, `uLayerDepth[${i}]`);
      if (i < count) {
        gl.activeTexture(gl.TEXTURE3 + i);
        gl.bindTexture(gl.TEXTURE_2D, assets.layers[i].tex);
        gl.uniform1i(loc, 3 + i);
        gl.uniform1f(dloc, assets.layers[i].depth);
      } else {
        gl.uniform1f(dloc, 0.0);
      }
    }
    gl.uniform1i(uWarp.uLayerCount, count);
    gl.uniform2f(uWarp.uPan, state.camera.pan[0] * 0.25, state.camera.pan[1] * 0.25);
    gl.uniform1f(uWarp.uParallax, state.params.parallax);
    gl.uniform1f(uWarp.uFlowStrength, state.params.flow * 0.55);
    gl.uniform1f(uWarp.uTime, time);
    gl.uniform1f(uWarp.uFlowScale, assets.manifest.flow.scale || 1);
    gl.uniform2f(uWarp.uImageSize, assets.manifest.width, assets.manifest.height);
    gl.uniform2f(uWarp.uCameraCenter, view.center[0], view.center[1]);
    gl.uniform2f(uWarp.uScale, view.scale[0], view.scale[1]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function drawBrush(time) {
    if (!assets.strokes || state.brushAlpha <= 0.01) return;
    gl.useProgram(progBrush);
    gl.bindVertexArray(strokeVAO);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, assets.flow);
    gl.uniform1i(uBrush.uFlow, 0);
    gl.uniform2f(uBrush.uImageSize, assets.manifest.width, assets.manifest.height);
    gl.uniform2f(uBrush.uCameraCenter, view.center[0], view.center[1]);
    gl.uniform2f(uBrush.uScale, view.scale[0], view.scale[1]);
    gl.uniform2f(uBrush.uPan, state.camera.pan[0] * 0.25, state.camera.pan[1] * 0.25);
    gl.uniform1f(uBrush.uParallax, state.params.parallax);
    gl.uniform1f(uBrush.uStrokeMotion, state.params.stroke);
    gl.uniform1f(uBrush.uTime, time);
    gl.uniform1f(uBrush.uTimeScale, state.params.speed);
    gl.uniform1f(uBrush.uFlowScale, assets.manifest.flow.scale || 1);
    gl.uniform1f(uBrush.uBrushScale, state.params.brush);
    gl.uniform1f(uBrush.uGlobalAlpha, state.brushAlpha);
    for (let i = 0; i < 6; i++) {
      const dloc = gl.getUniformLocation(progBrush, `uLayerDepth[${i}]`);
      gl.uniform1f(dloc, i < assets.layers.length ? assets.layers[i].depth : 0.0);
    }
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, assets.strokeCount);
  }

  function draw(nowSec) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clearColor(0.05, 0.058, 0.078, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (state.mode === "brush") {
      drawFlatQuad(assets.original, 0.42, nowSec);
      drawBrush(nowSec);
    } else {
      drawWarp(nowSec);
    }
  }

  // ---------------------------------------------------------------------------
  // animation loop
  // ---------------------------------------------------------------------------
  let lastFrame = performance.now();

  function tick(now) {
    const dt = Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;

    // fps
    state.fps.ema = lerp(state.fps.ema, 1 / Math.max(dt, 1e-4), 0.08);
    if (now - state.fps.last > 500) {
      $("fps").textContent = `${Math.round(state.fps.ema)} fps`;
      state.fps.last = now;
    }

    if (!state.paused) {
      state.time += dt * state.params.speed;
      state.brushAlpha = Math.min(1, state.brushAlpha + dt * 1.8);
    }

    // auto-pan + pointer + drag spring
    const m = assets.manifest ? Math.max(assets.manifest.width, assets.manifest.height) : 768;
    const amp = 0.05 * m * state.params.autoPan;
    const t = state.time * 0.35;
    const auto = [Math.sin(t * 0.9) * amp, Math.sin(t * 0.7 + 1.6) * amp * 0.7];
    const pointer = [state.camera.pointer[0] * m * 0.03, state.camera.pointer[1] * m * 0.03];
    state.camera.targetPan = [auto[0] + pointer[0] + state.camera.drag[0],
                              auto[1] + pointer[1] + state.camera.drag[1]];
    state.camera.pan = [
      lerp(state.camera.pan[0], state.camera.targetPan[0], 1 - Math.pow(0.001, dt)),
      lerp(state.camera.pan[1], state.camera.targetPan[1], 1 - Math.pow(0.001, dt)),
    ];
    state.camera.zoom = lerp(state.camera.zoom, state.camera.targetZoom, 1 - Math.pow(0.0005, dt));

    resize();
    updateView();
    draw(state.time);
    requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------------------
  // UI wiring
  // ---------------------------------------------------------------------------
  function bindSlider(id, key, out, fmt = (v) => v.toFixed(2)) {
    const el = $(id);
    el.addEventListener("input", () => {
      state.params[key] = parseFloat(el.value);
      $(out).textContent = fmt(state.params[key]);
    });
  }

  function setMode(mode) {
    state.mode = mode;
    $("modeBrush").classList.toggle("active", mode === "brush");
    $("modeWarp").classList.toggle("active", mode === "warp");
    state.brushAlpha = 0;    // fade strokes in again
  }

  function bindUI() {
    bindSlider("parallax", "parallax", "parallaxOut");
    bindSlider("flow", "flow", "flowOut");
    bindSlider("stroke", "stroke", "strokeOut");
    bindSlider("speed", "speed", "speedOut");
    bindSlider("brush", "brush", "brushOut");
    bindSlider("autoPan", "autoPan", "panOut");

    $("modeBrush").addEventListener("click", () => setMode("brush"));
    $("modeWarp").addEventListener("click", () => setMode("warp"));

    $("pause").addEventListener("click", () => {
      state.paused = !state.paused;
      $("pause").textContent = state.paused ? "Play" : "Pause";
    });
    $("reset").addEventListener("click", () => {
      state.camera.drag = [0, 0];
      state.camera.targetZoom = state.camera.zoom = 1;
      state.time = 0;
    });
    $("shot").addEventListener("click", screenshot);
    $("panelToggle").addEventListener("click", () => $("panel").classList.toggle("collapsed"));

    // pointer interactions
    canvas.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      state.camera.pointer = [
        (e.clientX - rect.left) / rect.width - 0.5,
        (e.clientY - rect.top) / rect.height - 0.5,
      ];
      if (state.camera.dragStart) {
        const dx = e.clientX - state.camera.dragStart[0];
        const dy = e.clientY - state.camera.dragStart[1];
        const m = Math.max(assets.manifest.width, assets.manifest.height);
        state.camera.drag = [state.camera.dragBase[0] - dx * m / 900,
                             state.camera.dragBase[1] - dy * m / 900];
      }
    });
    canvas.addEventListener("pointerdown", (e) => {
      state.camera.dragStart = [e.clientX, e.clientY];
      state.camera.dragBase = [...state.camera.drag];
      canvas.classList.add("dragging");
      canvas.setPointerCapture(e.pointerId);
    });
    const endDrag = () => {
      state.camera.dragStart = null;
      canvas.classList.remove("dragging");
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("pointerleave", () => { state.camera.pointer = [0, 0]; });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const k = Math.exp(-e.deltaY * 0.0012);
      state.camera.targetZoom = clamp(state.camera.targetZoom * k, 0.5, 3.2);
    }, { passive: false });
    canvas.addEventListener("dblclick", () => {
      state.camera.drag = [0, 0];
      state.camera.targetZoom = 1;
    });

    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
      switch (e.key) {
        case " ": e.preventDefault(); $("pause").click(); break;
        case "r": case "R": $("reset").click(); break;
        case "s": case "S": screenshot(); break;
        case "1": setMode("brush"); break;
        case "2": setMode("warp"); break;
        case "h": case "H": $("panel").classList.toggle("collapsed"); break;
        default: break;
      }
    });

    window.addEventListener("resize", resize);
  }

  function screenshot() {
    // draw synchronously so the buffer is fresh, then export
    draw(state.time);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `vangogh_${state.scene}_${state.mode}_${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    }, "image/png");
  }

  // ---------------------------------------------------------------------------
  // overlay helpers
  // ---------------------------------------------------------------------------
  function showOverlay(title, msg) {
    $("overlayTitle").textContent = title;
    $("overlayMsg").textContent = msg || "";
    $("overlay").classList.remove("hidden");
  }
  function hideOverlay() { $("overlay").classList.add("hidden"); }

  function webglFallback(msg) {
    $("fallbackImg").src = state.scenes.length && state.scenes[0].thumbnail
      ? state.scenes[0].thumbnail : "";
    $("fallback").classList.remove("hidden");
    $("fallback").querySelector("p").textContent =
      `WebGL2 is unavailable (${msg || "context creation failed"}) — showing the original painting instead.`;
  }

  // ---------------------------------------------------------------------------
  // boot
  // ---------------------------------------------------------------------------
  async function boot() {
    bindUI();
    if (!initGL()) {
      try {
        state.scenes = await fetchJSON("assets/scenes.json");
      } catch (err) { /* no assets at all */ }
      webglFallback();
      return;
    }

    try {
      const [flatVs, flatFs, warpVs, warpFs, brushVs, brushFs] = await Promise.all([
        loadShaderSource("flat.vert"),
        loadShaderSource("flat.frag"),
        loadShaderSource("flat.vert"),
        loadShaderSource("depth_parallax.frag"),
        loadShaderSource("brush.vert"),
        loadShaderSource("brush.frag"),
      ]);
      progFlat = link(flatVs, flatFs, "flat");
      progWarp = link(warpVs, warpFs, "warp");
      progBrush = link(brushVs, brushFs, "brush");
      uFlat = uniforms(progFlat, ["uTex", "uDim", "uTime", "uCameraCenter", "uScale", "uImageSize"]);
      uWarp = uniforms(progWarp, ["uOriginal", "uDepth", "uFlow", "uLayerCount", "uPan",
                                  "uParallax", "uFlowStrength", "uTime", "uFlowScale", "uImageSize",
                                  "uCameraCenter", "uScale"]);
      uBrush = uniforms(progBrush, ["uFlow", "uImageSize", "uCameraCenter", "uScale", "uPan",
                                    "uParallax", "uStrokeMotion", "uTime", "uTimeScale",
                                    "uFlowScale", "uBrushScale", "uGlobalAlpha"]);
    } catch (err) {
      showOverlay("Renderer error", String(err.message || err));
      console.error(err);
      return;
    }

    try {
      state.scenes = await fetchJSON("assets/scenes.json");
    } catch (err) {
      showOverlay("No scenes found", "Run the pipeline first:  python -m inference.run_pipeline");
      return;
    }

    const select = $("sceneSelect");
    select.innerHTML = "";
    for (const s of state.scenes) {
      const opt = document.createElement("option");
      opt.value = s.scene;
      opt.textContent = `${s.title}  (${s.layers} layers · ${s.strokes.toLocaleString()} strokes)`;
      select.appendChild(opt);
    }
    select.disabled = false;
    select.addEventListener("change", async () => {
      state.scene = select.value;
      await loadScene(state.scene);
    });

    state.scene = state.scenes[0].scene;
    select.value = state.scene;

    try {
      await loadScene(state.scene);
    } catch (err) {
      showOverlay("Scene failed to load", String(err.message || err));
      console.error(err);
      return;
    }

    lastFrame = performance.now();
    requestAnimationFrame(tick);
  }

  boot().catch((err) => {
    console.error(err);
    showOverlay("Fatal error", String(err.message || err));
  });
})();

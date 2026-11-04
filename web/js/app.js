/**
 * Physics-Informed Relative Depth Estimation Web Application
 * Handles split slider, point-to-point relative distance probing, 3D WebGL mesh, and uploads.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dualGrid = document.getElementById("dual-grid");
  const splitViewport = document.getElementById("split-viewport");
  const splitClip = document.getElementById("split-clip");
  const splitHandle = document.getElementById("split-handle");
  const imgRgb = document.getElementById("img-rgb");
  const imgDepth = document.getElementById("img-depth");
  const imgRgbSplit = document.getElementById("img-rgb-split");
  const imgDepthSplit = document.getElementById("img-depth-split");
  const boxRgb = document.getElementById("box-rgb");
  const boxDepth = document.getElementById("box-depth");
  const loadingOverlay = document.getElementById("loading-overlay");

  const tabSide = document.getElementById("tab-side");
  const tabSplit = document.getElementById("tab-split");
  const tabDepth = document.getElementById("tab-depth");
  const tabRgb = document.getElementById("tab-rgb");
  const tab3d = document.getElementById("tab-3d");
  const threejsContainer = document.getElementById("threejs-container");

  const modelSelect = document.getElementById("model-select");
  const colormapSelect = document.getElementById("colormap-select");
  const activeColormapLabel = document.getElementById("active-colormap-label");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const samplesContainer = document.getElementById("samples-container");

  const probeReadout = document.getElementById("probe-readout");
  const probeCoord = document.getElementById("probe-coord");
  const probeVal = document.getElementById("probe-val");

  const pinARgb = document.getElementById("pin-a-rgb");
  const pinBRgb = document.getElementById("pin-b-rgb");
  const pinADepth = document.getElementById("pin-a-depth");
  const pinBDepth = document.getElementById("pin-b-depth");
  const pinASplit = document.getElementById("pin-a-split");
  const pinBSplit = document.getElementById("pin-b-split");
  const crosshairRgb = document.getElementById("crosshair-rgb");
  const crosshairDepth = document.getElementById("crosshair-depth");
  const btnPinMode = document.getElementById("btn-pin-mode");
  const btnResetPins = document.getElementById("btn-reset-pins");

  const compHeadline = document.getElementById("comp-headline");
  const compSummary = document.getElementById("comp-summary");
  const valPointA = document.getElementById("val-point-a");
  const valPointB = document.getElementById("val-point-b");
  const valDelta = document.getElementById("val-delta");
  const valConfidence = document.getElementById("val-confidence");

  const airlightSwatch = document.getElementById("airlight-swatch");
  const airlightVal = document.getElementById("airlight-val");
  const betaVal = document.getElementById("beta-val");
  const rangeVal = document.getElementById("range-val");
  const deviceBadge = document.getElementById("device-badge");

  // Depth Scale Bar & Color Legend Elements
  const depthScaleGradient = document.getElementById("depth-scale-gradient");
  const scalePointer = document.getElementById("scale-pointer");
  const pointerBubble = document.getElementById("pointer-bubble");
  const scaleRoleLeft = document.getElementById("scale-role-left");
  const scaleRoleRight = document.getElementById("scale-role-right");
  const scaleTick0 = document.getElementById("scale-tick-0");
  const scaleTick1 = document.getElementById("scale-tick-1");
  const probeMiniGradient = document.getElementById("probe-mini-gradient");
  const probeMiniPointer = document.getElementById("probe-mini-pointer");
  const miniLabelNear = document.getElementById("mini-label-near");
  const miniLabelFar = document.getElementById("mini-label-far");
  const sceneDimBadge = document.getElementById("scene-dim-badge");

  // Colormap Gradients for Interactive Scale Bar
  const COLORMAP_GRADIENTS = {
    thermal_ironbow: {
      normal: "linear-gradient(to right, #ffffff 0%, #feffa5 8%, #f59e0b 20%, #fb923c 38%, #e11d48 55%, #7c3aed 72%, #1e1b4b 88%, #020617 100%)",
      inverted: "linear-gradient(to right, #020617 0%, #1e1b4b 12%, #7c3aed 28%, #e11d48 45%, #fb923c 62%, #f59e0b 80%, #feffa5 92%, #ffffff 100%)",
    },
    thermal_rainbow: {
      normal: "linear-gradient(to right, #ffffff 0%, #ef4444 14%, #f97316 28%, #eab308 42%, #22c55e 58%, #06b6d4 72%, #2563eb 86%, #020617 100%)",
      inverted: "linear-gradient(to right, #020617 0%, #2563eb 14%, #06b6d4 28%, #22c55e 42%, #eab308 58%, #f97316 72%, #ef4444 86%, #ffffff 100%)",
    },
    distinct: {
      normal: "linear-gradient(to right, #ffff00 0%, #ff9900 15%, #f22e1f 30%, #d91a8c 45%, #7326d9 60%, #1f80f2 75%, #0d2e8c 90%, #000005 100%)",
      inverted: "linear-gradient(to right, #000005 0%, #0d2e8c 10%, #1f80f2 25%, #7326d9 40%, #d91a8c 55%, #f22e1f 70%, #ff9900 85%, #ffff00 100%)",
    },
    das_icvgip: {
      normal: "linear-gradient(to right, #ffff00 0%, #ff8c00 20%, #d91a40 45%, #2659e6 70%, #0f144d 85%, #000005 100%)",
      inverted: "linear-gradient(to right, #000005 0%, #0f144d 15%, #2659e6 30%, #d91a40 55%, #ff8c00 80%, #ffff00 100%)",
    },
    turbo: {
      normal: "linear-gradient(to right, #7a0403 0%, #d93806 18%, #f68925 36%, #b5de2b 54%, #35b779 68%, #31688e 82%, #440154 100%)",
      inverted: "linear-gradient(to right, #440154 0%, #31688e 18%, #35b779 32%, #b5de2b 46%, #f68925 64%, #d93806 82%, #7a0403 100%)",
    },
    inferno: {
      normal: "linear-gradient(to right, #fcffa4 0%, #f98e09 30%, #bc3754 55%, #57106e 80%, #000004 100%)",
      inverted: "linear-gradient(to right, #000004 0%, #57106e 20%, #bc3754 45%, #f98e09 70%, #fcffa4 100%)",
    },
    viridis: {
      normal: "linear-gradient(to right, #fde725 0%, #5ec962 30%, #21918c 55%, #3b528b 80%, #440154 100%)",
      inverted: "linear-gradient(to right, #440154 0%, #3b528b 20%, #21918c 45%, #5ec962 70%, #fde725 100%)",
    },
    magma: {
      normal: "linear-gradient(to right, #fcfdbf 0%, #fe9f6d 30%, #de4968 55%, #862781 80%, #000004 100%)",
      inverted: "linear-gradient(to right, #000004 0%, #862781 20%, #de4968 45%, #fe9f6d 70%, #fcfdbf 100%)",
    },
    grayscale: {
      normal: "linear-gradient(to right, #ffffff 0%, #000000 100%)",
      inverted: "linear-gradient(to right, #000000 0%, #ffffff 100%)",
    },
  };

  function updateDepthScaleBar() {
    const palKey = colormapSelect ? colormapSelect.value : "thermal_ironbow";
    const palData = COLORMAP_GRADIENTS[palKey] || COLORMAP_GRADIENTS.thermal_ironbow;
    const grad = palData.normal;

    if (depthScaleGradient) {
      depthScaleGradient.style.background = grad;
    }
    if (probeMiniGradient) {
      probeMiniGradient.style.background = grad;
    }

    if (scaleRoleLeft) {
      scaleRoleLeft.textContent = "CLOSE (Foreground)";
    }
    if (scaleRoleRight) {
      scaleRoleRight.textContent = "DISTANT (Sky / Far)";
    }
    if (scaleTick0) {
      scaleTick0.textContent = "0.00 (Near)";
    }
    if (scaleTick1) {
      scaleTick1.textContent = "1.00 (Far)";
    }
    if (miniLabelNear) {
      miniLabelNear.textContent = "Near";
    }
    if (miniLabelFar) {
      miniLabelFar.textContent = "Far";
    }
  }

  // Application State
  let currentSampleId = "garden";
  let currentFile = null;
  let currentData = null;
  let isDraggingSplit = false;
  let activeTab = "side"; // 'side' (default), 'split', 'depth', 'rgb', '3d'
  let pinMode = true;
  let points = []; // [{x, y, relX, relY, depth}]

  // Three.js State
  let scene, camera, renderer, controls, mesh3d;

  // 1. Fetch Backend Health
  fetch("/health")
    .then((r) => r.json())
    .then((data) => {
      deviceBadge.textContent = `${data.device.toUpperCase()} • ${data.backbone.toUpperCase()}`;
    })
    .catch(() => {
      deviceBadge.textContent = "Offline";
    });

  // 2. Fetch Sample Gallery
  fetch("/api/samples")
    .then((r) => r.json())
    .then((samples) => {
      samplesContainer.innerHTML = "";
      samples.forEach((sample) => {
        const thumb = document.createElement("div");
        thumb.className = `sample-thumbnail ${sample.id === currentSampleId ? "active" : ""}`;
        thumb.title = sample.title;
        thumb.innerHTML = `<img src="${sample.url}" alt="${sample.title}">`;
        thumb.addEventListener("click", () => {
          document.querySelectorAll(".sample-thumbnail").forEach((el) => el.classList.remove("active"));
          thumb.classList.add("active");
          currentSampleId = sample.id;
          currentFile = null;
          resetPins();
          loadPrediction();
        });
        samplesContainer.appendChild(thumb);
      });
    });

  // 3. Load Prediction
  function loadPrediction() {
    loadingOverlay.style.display = "flex";
    const formData = new FormData();

    if (currentFile) {
      formData.append("file", currentFile);
    } else {
      formData.append("sample_id", currentSampleId);
    }

    formData.append("colormap", colormapSelect ? colormapSelect.value : "thermal_ironbow");
    formData.append("invert", "true");
    if (modelSelect) {
      formData.append("model_type", modelSelect.value);
    }
    // Optimal default parameters for clear thermal camera visualization
    formData.append("gamma", "1.35");
    formData.append("show_contours", "false");
    formData.append("use_clahe", "false");
    formData.append("msx_detail", "true");
    formData.append("msx_strength", "0.55");

    fetch("/api/predict", {
      method: "POST",
      body: formData,
    })
      .then((r) => {
        if (!r.ok) throw new Error("Inference failed");
        return r.json();
      })
      .then((data) => {
        currentData = data;
        imgRgb.src = data.original_image;
        imgDepth.src = data.depth_map_png;
        if (imgRgbSplit) imgRgbSplit.src = data.original_image;
        if (imgDepthSplit) imgDepthSplit.src = data.depth_map_png;
        if (activeColormapLabel) {
          const selOpt = colormapSelect.options[colormapSelect.selectedIndex];
          activeColormapLabel.textContent = selOpt ? selOpt.text.split("(")[0].trim() : "Multi-Hue";
        }

        // Update Physics Parameters Card
        const air = data.physics.airlight;
        const r = Math.round(air[0] * 255);
        const g = Math.round(air[1] * 255);
        const b = Math.round(air[2] * 255);
        airlightSwatch.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        airlightVal.textContent = `[${air[0].toFixed(2)}, ${air[1].toFixed(2)}, ${air[2].toFixed(2)}]`;
        betaVal.textContent = data.physics.scattering_beta.toFixed(2);
        const dasBetaEl = document.getElementById("das-beta-val");
        if (dasBetaEl && data.physics.das_beta) {
          dasBetaEl.textContent = data.physics.das_beta.toFixed(2);
        }
        rangeVal.textContent = `[${data.statistics.min.toFixed(2)} – ${data.statistics.max.toFixed(2)}]`;

        if (sceneDimBadge && data.image_size) {
          sceneDimBadge.textContent = `${data.image_size[0]} × ${data.image_size[1]} px`;
        }
        updateDepthScaleBar();

        setSplitPosition(50);

        if (activeTab === "3d") {
          initOrUpdate3D();
        }
      })
      .catch((err) => {
        console.error(err);
        alert("Error estimating depth: " + err.message);
      })
      .finally(() => {
        loadingOverlay.style.display = "none";
      });
  }

  // Initial setup & prediction load
  updateDepthScaleBar();
  setTab("side");
  loadPrediction();

  // 4. Split Slider Interaction
  function setSplitPosition(percent) {
    const p = Math.max(0, Math.min(100, percent));
    splitClip.style.width = `${p}%`;
    splitHandle.style.left = `${p}%`;
  }

  if (splitViewport) {
    splitViewport.addEventListener("mousedown", (e) => {
      if (activeTab !== "split") return;
      const rect = splitViewport.getBoundingClientRect();
      const handleX = rect.left + (rect.width * parseFloat(splitHandle.style.left || 50)) / 100;
      if (Math.abs(e.clientX - handleX) < 25) {
        isDraggingSplit = true;
        e.preventDefault();
      }
    });

    splitViewport.addEventListener("mousemove", (e) => {
      if (activeTab !== "split" && activeTab !== "depth" && activeTab !== "rgb") return;
      const rect = splitViewport.getBoundingClientRect();
      const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      handleProbeHover(relX, relY, false);
    });

    splitViewport.addEventListener("click", (e) => {
      if (!currentData || isDraggingSplit) return;
      const rect = splitViewport.getBoundingClientRect();
      const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      handlePointSelection(relX, relY);
    });

    splitViewport.addEventListener("mouseleave", handleProbeLeave);
  }

  window.addEventListener("mousemove", (e) => {
    if (!isDraggingSplit || activeTab !== "split" || !splitViewport) return;
    const rect = splitViewport.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = (x / rect.width) * 100;
    setSplitPosition(percent);
  });

  window.addEventListener("mouseup", () => {
    isDraggingSplit = false;
  });

  // 5. Hover Depth Probing & Point Selection
  function handleProbeHover(relX, relY, showCrosshairs = true) {
    if (!currentData || activeTab === "3d") return;

    const grid = currentData.probe_grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const rIdx = Math.min(rows - 1, Math.floor(relY * rows));
    const cIdx = Math.min(cols - 1, Math.floor(relX * cols));
    const dVal = grid[rIdx][cIdx];

    const pixelX = Math.round(relX * currentData.image_size[0]);
    const pixelY = Math.round(relY * currentData.image_size[1]);

    probeCoord.textContent = `${pixelX}, ${pixelY}`;
    probeVal.textContent = `${dVal.toFixed(2)} (${dVal < 0.35 ? "Near" : dVal < 0.7 ? "Midground" : "Far / Sky"})`;

    if (showCrosshairs && crosshairRgb && crosshairDepth) {
      const xp = `${relX * 100}%`;
      const yp = `${relY * 100}%`;
      crosshairRgb.style.display = "block";
      crosshairRgb.style.left = xp;
      crosshairRgb.style.top = yp;

      crosshairDepth.style.display = "block";
      crosshairDepth.style.left = xp;
      crosshairDepth.style.top = yp;
    }

    // Update dynamic scale pointers
    const normP = Math.max(0, Math.min(100, dVal * 100));
    if (scalePointer && pointerBubble) {
      scalePointer.style.display = "block";
      scalePointer.style.left = `${normP}%`;
      pointerBubble.textContent = dVal.toFixed(2);
    }
    if (probeMiniPointer) {
      probeMiniPointer.style.display = "block";
      probeMiniPointer.style.left = `${normP}%`;
    }
  }

  function handleProbeLeave() {
    if (crosshairRgb) crosshairRgb.style.display = "none";
    if (crosshairDepth) crosshairDepth.style.display = "none";
    if (scalePointer) scalePointer.style.display = "none";
    if (probeMiniPointer) probeMiniPointer.style.display = "none";
  }

  function handlePointSelection(relX, relY) {
    if (!currentData || activeTab === "3d" || isDraggingSplit) return;

    const grid = currentData.probe_grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const rIdx = Math.min(rows - 1, Math.floor(relY * rows));
    const cIdx = Math.min(cols - 1, Math.floor(relX * cols));
    const depth = grid[rIdx][cIdx];

    if (points.length >= 2) {
      resetPins();
    }

    points.push({
      relX,
      relY,
      xPercent: relX * 100,
      yPercent: relY * 100,
      depth,
    });

    renderPins();
    evaluateRelativeOrdering();
  }

  function attachBoxListeners(box) {
    if (!box) return;
    box.addEventListener("mousemove", (e) => {
      const rect = box.getBoundingClientRect();
      const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      handleProbeHover(relX, relY, true);
    });

    box.addEventListener("mouseleave", handleProbeLeave);

    box.addEventListener("click", (e) => {
      const rect = box.getBoundingClientRect();
      const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      handlePointSelection(relX, relY);
    });

    splitViewport.addEventListener("mouseleave", handleProbeLeave);
  }

  attachBoxListeners(boxRgb);
  attachBoxListeners(boxDepth);

  function renderPins() {
    const allPinA = [pinARgb, pinADepth, pinASplit].filter(Boolean);
    const allPinB = [pinBRgb, pinBDepth, pinBSplit].filter(Boolean);

    if (points.length >= 1) {
      allPinA.forEach((p) => {
        p.style.display = "flex";
        p.style.left = `${points[0].xPercent}%`;
        p.style.top = `${points[0].yPercent}%`;
      });
      valPointA.textContent = points[0].depth.toFixed(2);
    } else {
      allPinA.forEach((p) => (p.style.display = "none"));
      valPointA.textContent = "-";
    }

    if (points.length >= 2) {
      allPinB.forEach((p) => {
        p.style.display = "flex";
        p.style.left = `${points[1].xPercent}%`;
        p.style.top = `${points[1].yPercent}%`;
      });
      valPointB.textContent = points[1].depth.toFixed(2);
    } else {
      allPinB.forEach((p) => (p.style.display = "none"));
      valPointB.textContent = "-";
    }
  }

  function resetPins() {
    points = [];
    renderPins();
    compHeadline.textContent = "Select Points A & B";
    compSummary.textContent = "Click anywhere on the image to place Point A, then click another object to place Point B.";
    valDelta.textContent = "-";
    valConfidence.textContent = "-";
  }

  btnResetPins.addEventListener("click", resetPins);

  function evaluateRelativeOrdering() {
    if (points.length < 2) {
      compHeadline.textContent = points.length === 1 ? "Point A Set — Click Point B" : "Select Points A & B";
      return;
    }

    const payload = {
      point_a: [points[0].relX, points[0].relY],
      point_b: [points[1].relX, points[1].relY],
      depth_grid: currentData.probe_grid,
    };

    fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((res) => {
        compHeadline.textContent = res.closer;
        compSummary.textContent = res.summary;
        valDelta.textContent = res.delta.toFixed(3);
        valConfidence.textContent = `${res.confidence_score}%`;
      });
  }

  // 6. View Tabs
  function setTab(tab) {
    activeTab = tab;
    [tabSide, tabSplit, tabDepth, tabRgb, tab3d].filter(Boolean).forEach((t) => t.classList.remove("active"));

    if (tab === "side") {
      if (tabSide) tabSide.classList.add("active");
      if (dualGrid) dualGrid.style.display = "grid";
      if (splitViewport) splitViewport.style.display = "none";
      if (threejsContainer) threejsContainer.style.display = "none";
      if (probeReadout) probeReadout.style.display = "flex";
    } else if (tab === "split") {
      if (tabSplit) tabSplit.classList.add("active");
      if (dualGrid) dualGrid.style.display = "none";
      if (splitViewport) splitViewport.style.display = "block";
      if (splitClip) splitClip.style.display = "block";
      if (splitHandle) splitHandle.style.display = "block";
      if (threejsContainer) threejsContainer.style.display = "none";
      if (probeReadout) probeReadout.style.display = "flex";
      setSplitPosition(50);
    } else if (tab === "depth") {
      if (tabDepth) tabDepth.classList.add("active");
      if (dualGrid) dualGrid.style.display = "none";
      if (splitViewport) splitViewport.style.display = "block";
      if (splitClip) splitClip.style.display = "block";
      if (splitHandle) splitHandle.style.display = "none";
      if (threejsContainer) threejsContainer.style.display = "none";
      if (probeReadout) probeReadout.style.display = "flex";
      setSplitPosition(100);
    } else if (tab === "rgb") {
      if (tabRgb) tabRgb.classList.add("active");
      if (dualGrid) dualGrid.style.display = "none";
      if (splitViewport) splitViewport.style.display = "block";
      if (splitClip) splitClip.style.display = "block";
      if (splitHandle) splitHandle.style.display = "none";
      if (threejsContainer) threejsContainer.style.display = "none";
      if (probeReadout) probeReadout.style.display = "flex";
      setSplitPosition(0);
    } else if (tab === "3d") {
      if (tab3d) tab3d.classList.add("active");
      if (dualGrid) dualGrid.style.display = "none";
      if (splitViewport) splitViewport.style.display = "none";
      if (threejsContainer) threejsContainer.style.display = "block";
      if (probeReadout) probeReadout.style.display = "none";
      initOrUpdate3D();
    }
  }

  if (tabSide) tabSide.addEventListener("click", () => setTab("side"));
  if (tabSplit) tabSplit.addEventListener("click", () => setTab("split"));
  if (tabDepth) tabDepth.addEventListener("click", () => setTab("depth"));
  if (tabRgb) tabRgb.addEventListener("click", () => setTab("rgb"));
  if (tab3d) tab3d.addEventListener("click", () => setTab("3d"));

  // 7. Model & Palette Controls
  if (modelSelect) modelSelect.addEventListener("change", loadPrediction);
  if (colormapSelect) {
    colormapSelect.addEventListener("change", () => {
      updateDepthScaleBar();
      loadPrediction();
    });
  }

  // 8. File Upload
  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      currentFile = e.target.files[0];
      currentSampleId = null;
      document.querySelectorAll(".sample-thumbnail").forEach((el) => el.classList.remove("active"));
      resetPins();
      loadPrediction();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent-blue)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "var(--border-medium)";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--border-medium)";
    if (e.dataTransfer.files.length > 0) {
      currentFile = e.dataTransfer.files[0];
      currentSampleId = null;
      document.querySelectorAll(".sample-thumbnail").forEach((el) => el.classList.remove("active"));
      resetPins();
      loadPrediction();
    }
  });

  // 9. Interactive Three.js 3D Surface / Point Cloud View
  function initOrUpdate3D() {
    if (!currentData) return;

    const width = threejsContainer.clientWidth || 800;
    const height = threejsContainer.clientHeight || 500;

    if (!scene) {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf8fafc);

      camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
      camera.position.set(0, -1.8, 2.2);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      threejsContainer.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
      scene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
      dirLight.position.set(1, 2, 3);
      scene.add(dirLight);

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }
      animate();

      window.addEventListener("resize", () => {
        if (renderer && camera && threejsContainer.style.display !== "none") {
          const w = threejsContainer.clientWidth || 800;
          const h = threejsContainer.clientHeight || 500;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      });
    }

    // Fetch 3D point cloud from backend
    const formData = new FormData();
    if (currentFile) {
      formData.append("file", currentFile);
    } else {
      formData.append("sample_id", currentSampleId);
    }

    fetch("/api/mesh-3d", {
      method: "POST",
      body: formData,
    })
      .then((r) => r.json())
      .then((meshData) => {
        if (mesh3d) {
          scene.remove(mesh3d);
          mesh3d.geometry.dispose();
          mesh3d.material.dispose();
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(meshData.positions, 3));
        geometry.setAttribute("color", new THREE.Float32BufferAttribute(meshData.colors, 3));

        const material = new THREE.PointsMaterial({
          size: 0.035,
          vertexColors: true,
          sizeAttenuation: true,
        });

        mesh3d = new THREE.Points(geometry, material);
        mesh3d.rotation.x = -Math.PI / 4; // subtle tilt for natural perspective
        scene.add(mesh3d);
      })
      .catch((err) => console.error("3D Mesh error:", err));
  }

  // 12. Code Walkthrough Tabs & Copy Functionality
  const codeTabBtns = document.querySelectorAll(".code-tab-btn");
  const codeTabPanels = document.querySelectorAll(".code-tab-panel");

  codeTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      codeTabBtns.forEach((b) => b.classList.remove("active"));
      codeTabPanels.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });

  const copyCodeBtns = document.querySelectorAll(".copy-code-btn");
  copyCodeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetCodeId = btn.getAttribute("data-target");
      const codeElem = document.getElementById(targetCodeId);
      if (codeElem) {
        const textToCopy = codeElem.innerText || codeElem.textContent;
        navigator.clipboard.writeText(textToCopy).then(() => {
          const originalHtml = btn.innerHTML;
          btn.classList.add("copied");
          btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px;vertical-align:-1px;"><polyline points="20 6 9 17 4 12"/></svg>Copied`;
          setTimeout(() => {
            btn.classList.remove("copied");
            btn.innerHTML = originalHtml;
          }, 2000);
        }).catch((err) => {
          console.error("Failed to copy code:", err);
        });
      }
    });
  });
});


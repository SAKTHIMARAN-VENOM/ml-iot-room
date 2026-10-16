/**
 * dashboard.js
 * ============
 * Frontend controller for Smart Room Occupancy Detection System.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Sliders and Displays
    const inputTemp = document.getElementById('inputTemp');
    const inputHum = document.getElementById('inputHum');
    const inputLight = document.getElementById('inputLight');
    const dispTemp = document.getElementById('dispTemp');
    const dispHum = document.getElementById('dispHum');
    const dispLight = document.getElementById('dispLight');

    // DOM Elements - Sensor Cards
    const valTemp = document.getElementById('valTemp');
    const valHum = document.getElementById('valHum');
    const valLight = document.getElementById('valLight');
    const valPIR = document.getElementById('valPIR');
    const pirCardMeta = document.getElementById('pirCardMeta');

    // DOM Elements - Combined Decision
    const combinedBadge = document.getElementById('combinedBadge');
    const combinedIcon = document.getElementById('combinedIcon');
    const combinedStatusText = document.getElementById('combinedStatusText');
    const confidenceTag = document.getElementById('confidenceTag');
    const combinedReasoningText = document.getElementById('combinedReasoningText');

    // DOM Elements - Sub Cards
    const mlStatusText = document.getElementById('mlStatusText');
    const mlProbabilityText = document.getElementById('mlProbabilityText');
    const mlProbabilityBar = document.getElementById('mlProbabilityBar');
    const pirPinText = document.getElementById('pirPinText');
    const pirTimerText = document.getElementById('pirTimerText');
    const pirTimeoutMeta = document.getElementById('pirTimeoutMeta');

    // Controls
    const form = document.getElementById('predictionForm');
    const btnPulsePIR = document.getElementById('btnPulsePIR');
    const inputTimeout = document.getElementById('inputTimeout');
    const btnSaveTimeout = document.getElementById('btnSaveTimeout');
    const presetButtons = document.querySelectorAll('.preset-btn');

    // Modal
    const btnViewCM = document.getElementById('btnViewCM');
    const btnCloseModal = document.getElementById('btnCloseModal');
    const cmModal = document.getElementById('cmModal');

    // State Variables
    let lastMotionTimestamp = null;
    let currentTimeoutSeconds = 300;
    let trendsChart = null;

    // Presets Definition
    const PRESETS = {
        empty_night: { temp: 19.5, hum: 22.0, light: 0, pir: 0 },
        active_office: { temp: 23.8, hum: 26.5, light: 550, pir: 1 },
        seated_reading: { temp: 23.2, hum: 27.0, light: 480, pir: 0 },
        doorway_walk: { temp: 20.0, hum: 23.0, light: 10, pir: 1 }
    };

    // -------------------------------------------------------------------------
    // Slider Sync Listeners
    // -------------------------------------------------------------------------
    inputTemp.addEventListener('input', (e) => {
        dispTemp.textContent = `${parseFloat(e.target.value).toFixed(1)} °C`;
    });
    inputHum.addEventListener('input', (e) => {
        dispHum.textContent = `${parseFloat(e.target.value).toFixed(1)} %`;
    });
    inputLight.addEventListener('input', (e) => {
        dispLight.textContent = `${parseInt(e.target.value, 10)} Lux`;
    });

    // -------------------------------------------------------------------------
    // Preset Scenario Handlers
    // -------------------------------------------------------------------------
    presetButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const presetKey = btn.getAttribute('data-preset');
            const data = PRESETS[presetKey];
            if (data) {
                applyPreset(data);
                // Auto submit for rapid testing feedback
                submitPrediction();
            }
        });
    });

    function applyPreset(data) {
        inputTemp.value = data.temp;
        dispTemp.textContent = `${data.temp} °C`;

        inputHum.value = data.hum;
        dispHum.textContent = `${data.hum} %`;

        inputLight.value = data.light;
        dispLight.textContent = `${data.light} Lux`;

        if (data.pir === 1) {
            document.getElementById('pirHighRadio').checked = true;
        } else {
            document.getElementById('pirLowRadio').checked = true;
        }
    }

    // -------------------------------------------------------------------------
    // Initialize Chart.js
    // -------------------------------------------------------------------------
    function initChart() {
        const ctx = document.getElementById('trendsChart').getContext('2d');
        trendsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Occupancy Prob (%)',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        borderWidth: 2,
                        tension: 0.35,
                        fill: true,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Light Level (Lux)',
                        data: [],
                        borderColor: '#06b6d4',
                        borderWidth: 1.5,
                        tension: 0.35,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#64748b', font: { size: 10 } }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        position: 'left',
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: {
                            color: '#94a3b8',
                            callback: (v) => `${v}%`
                        }
                    },
                    y1: {
                        min: 0,
                        max: 1000,
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: {
                            color: '#06b6d4',
                            callback: (v) => `${v} lx`
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#cbd5e1', font: { size: 11 } }
                    }
                }
            }
        });
    }

    function appendChartData(timeStr, occProbPercent, lightLux) {
        if (!trendsChart) return;
        const maxPoints = 20;
        trendsChart.data.labels.push(timeStr);
        trendsChart.data.datasets[0].data.push(occProbPercent);
        trendsChart.data.datasets[1].data.push(lightLux);

        if (trendsChart.data.labels.length > maxPoints) {
            trendsChart.data.labels.shift();
            trendsChart.data.datasets[0].data.shift();
            trendsChart.data.datasets[1].data.shift();
        }
        trendsChart.update();
    }

    // -------------------------------------------------------------------------
    // Submit Prediction Request
    // -------------------------------------------------------------------------
    async function submitPrediction() {
        const temp = parseFloat(inputTemp.value);
        const hum = parseFloat(inputHum.value);
        const light = parseFloat(inputLight.value);
        const pirVal = parseInt(document.querySelector('input[name="pirState"]:checked').value, 10);

        const payload = {
            temperature: temp,
            humidity: hum,
            light: light,
            pir: pirVal,
            source: 'manual_test'
        };

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                alert(`Prediction error: ${errData.error || 'Server error'}`);
                return;
            }

            const data = await res.json();
            updateDashboardUI(data);

        } catch (err) {
            console.error('Failed to submit prediction:', err);
        }
    }

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        submitPrediction();
    });

    // -------------------------------------------------------------------------
    // Update Dashboard UI with Response
    // -------------------------------------------------------------------------
    function updateDashboardUI(data) {
        const inputs = data.inputs;
        const ml = data.ml_prediction;
        const pir = data.pir_status;
        const comb = data.combined_decision;

        // 1. Update Sensor Value Cards
        valTemp.textContent = inputs.temperature.toFixed(1);
        valHum.textContent = inputs.humidity.toFixed(1);
        valLight.textContent = inputs.light.toFixed(0);
        valPIR.textContent = inputs.pir_pin;
        pirCardMeta.textContent = inputs.pir_pin === 1 ? 'Pin: HIGH (Active)' : 'Pin: LOW (Idle)';

        // 2. Update Combined Decision Card
        combinedBadge.className = `main-badge ${comb.badge_variant}`;
        combinedStatusText.textContent = comb.combined_status;

        // Choose appropriate icon
        if (comb.badge_variant === 'success') {
            combinedIcon.textContent = '🟢';
        } else if (comb.badge_variant === 'warning') {
            combinedIcon.textContent = '🟡';
        } else if (comb.badge_variant === 'info') {
            combinedIcon.textContent = '🔵';
        } else {
            combinedIcon.textContent = '⚪';
        }

        confidenceTag.textContent = `Confidence: ${comb.confidence_level}`;
        combinedReasoningText.textContent = comb.decision_reasoning;

        // 3. Update Environmental ML Card
        const occPercent = (ml.occupied_probability * 100).toFixed(1);
        mlStatusText.textContent = `${ml.label}`;
        mlStatusText.style.color = ml.target === 1 ? '#34d399' : '#94a3b8';
        mlProbabilityText.textContent = `Prob: ${occPercent}%`;
        mlProbabilityBar.style.width = `${occPercent}%`;

        // 4. Update PIR Motion Card
        pirPinText.textContent = pir.pin_label;
        lastMotionTimestamp = pir.last_motion_timestamp;
        currentTimeoutSeconds = pir.timeout_seconds;
        updatePIRTimerDisplay();

        // 5. Append to Chart
        const shortTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        appendChartData(shortTime, parseFloat(occPercent), inputs.light);
    }

    // -------------------------------------------------------------------------
    // PIR Timer Dynamic Ticker
    // -------------------------------------------------------------------------
    function updatePIRTimerDisplay() {
        if (!lastMotionTimestamp) {
            pirTimerText.textContent = 'Last Motion: Never';
            return;
        }

        const nowSeconds = Date.now() / 1000;
        const elapsed = Math.max(0, Math.round(nowSeconds - lastMotionTimestamp));

        if (elapsed <= currentTimeoutSeconds) {
            pirTimerText.textContent = `Motion: ${elapsed}s ago (Active)`;
            pirTimerText.style.color = '#38bdf8';
        } else {
            pirTimerText.textContent = `Motion: ${elapsed}s ago (Expired)`;
            pirTimerText.style.color = '#64748b';
        }
    }

    // Tick PIR elapsed timer every second
    setInterval(updatePIRTimerDisplay, 1000);

    // -------------------------------------------------------------------------
    // Quick Pulse PIR Action
    // -------------------------------------------------------------------------
    btnPulsePIR.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/pir/motion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pin_state: 1, source: 'manual_pulse' })
            });
            const data = await res.json();
            if (data.status === 'success') {
                document.getElementById('pirHighRadio').checked = true;
                lastMotionTimestamp = data.pir_status.last_motion_timestamp;
                pirPinText.textContent = 'HIGH (Motion)';
                updatePIRTimerDisplay();
                // Run prediction with new motion state
                submitPrediction();
            }
        } catch (err) {
            console.error('Failed to pulse PIR:', err);
        }
    });

    // -------------------------------------------------------------------------
    // Save Configurable Timeout
    // -------------------------------------------------------------------------
    btnSaveTimeout.addEventListener('click', async () => {
        const timeoutVal = parseInt(inputTimeout.value, 10);
        if (isNaN(timeoutVal) || timeoutVal < 10) {
            alert('Timeout must be at least 10 seconds.');
            return;
        }

        try {
            const res = await fetch('/api/pir/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ timeout_seconds: timeoutVal })
            });
            const data = await res.json();
            if (data.status === 'success') {
                currentTimeoutSeconds = data.timeout_seconds;
                pirTimeoutMeta.textContent = `Hold Window: ${currentTimeoutSeconds}s timeout`;
                alert(`PIR Occupancy Timeout updated to ${currentTimeoutSeconds} seconds.`);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (err) {
            console.error('Failed to update timeout:', err);
        }
    });

    // -------------------------------------------------------------------------
    // Modal Management (Confusion Matrix)
    // -------------------------------------------------------------------------
    btnViewCM.addEventListener('click', () => {
        cmModal.classList.add('open');
    });

    btnCloseModal.addEventListener('click', () => {
        cmModal.classList.remove('open');
    });

    cmModal.addEventListener('click', (e) => {
        if (e.target === cmModal) {
            cmModal.classList.remove('open');
        }
    });

    // -------------------------------------------------------------------------
    // Initial Load & Diagnostics
    // -------------------------------------------------------------------------
    initChart();

    // Check health & fetch metrics
    fetch('/api/metrics')
        .then(res => res.json())
        .then(data => {
            if (data.test_metrics) {
                document.getElementById('metricAcc').textContent = `${(data.test_metrics.accuracy * 100).toFixed(2)}%`;
                document.getElementById('metricPrec').textContent = `${(data.test_metrics.precision_class1 * 100).toFixed(2)}%`;
                document.getElementById('metricRec').textContent = `${(data.test_metrics.recall_class1 * 100).toFixed(2)}%`;
                document.getElementById('metricF1').textContent = `${(data.test_metrics.f1_score_class1 * 100).toFixed(2)}%`;
            }
        })
        .catch(err => console.log('Could not fetch metrics:', err));

    // Execute first prediction with default values
    submitPrediction();

    // -------------------------------------------------------------------------
    // Automatic Live Telemetry Poller (Picks up ESP32 data stream)
    // -------------------------------------------------------------------------
    let lastSeenTimestamp = null;

    async function pollLatestData() {
        try {
            const res = await fetch('/api/latest');
            if (res.ok) {
                const data = await res.json();
                if (data && data.timestamp && data.timestamp !== lastSeenTimestamp) {
                    lastSeenTimestamp = data.timestamp;
                    
                    const hardwareStatusPill = document.getElementById('hardwareStatusPill');
                    const simulationBanner = document.getElementById('simulationBanner');
                    
                    if (data.input_mode === 'live_hardware') {
                        if (hardwareStatusPill) {
                            hardwareStatusPill.className = 'status-pill status-online';
                            hardwareStatusPill.innerHTML = '<span class="pulse-dot success"></span><span>ESP32 Hardware: Connected (Live Stream)</span>';
                        }
                        if (simulationBanner) {
                            simulationBanner.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
                            simulationBanner.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                            simulationBanner.style.color = '#a7f3d0';
                            simulationBanner.innerHTML = `<div class="alert-icon">🟢</div><div class="alert-content"><strong>Live Hardware Stream Active:</strong> Receiving live sensor telemetry from ESP32 (DHT11, LDR, PIR). Last packet received at <strong>${data.timestamp}</strong>.</div>`;
                        }
                    }
                    
                    updateDashboardUI(data);
                }
            }
        } catch (e) {
            // Silently ignore network interruptions during poll
        }
    }

    // Poll for fresh ESP32 readings every 2.5 seconds
    setInterval(pollLatestData, 2500);
});

// MultiProcess Dashboard JavaScript

let baseUrl = "";
let statusData = null;
let autoRefreshTimer = null;

// Initialize after DOM is loaded
window.addEventListener("DOMContentLoaded", () => {
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);

    // Base URL is the current page origin
    const protocol = window.location.protocol;
    const host = window.location.hostname;
    const port = window.location.port
        || (protocol === "https:" ? "443" : "80");
    baseUrl = protocol + "//" + host + ":" + port;

    // Refresh button
    document.getElementById("refreshAllBtn")
        .addEventListener("click", refreshAll);

    // Auto-refresh toggle
    document.getElementById("autoRefreshToggle")
        .addEventListener("change", toggleAutoRefresh);

    // Tab switching
    document.querySelectorAll(".nav-link").forEach(function(tab) {
        tab.addEventListener("shown.bs.tab", function() {
            if (statusData) {
                renderAll(statusData);
            }
        });
    });

    // JSON search
    document.getElementById("jsonSearchInput")
        .addEventListener("input", filterJson);

    // New tab refresh buttons
    document.getElementById("refreshPeriodicThreadsBtn")
        .addEventListener("click", refreshPeriodicThreads);
    document.getElementById("refreshThreadsBtn")
        .addEventListener("click", refreshThreads);
    document.getElementById("refreshEnvBtn")
        .addEventListener("click", refreshEnv);
    document.getElementById("refreshLogLevelBtn")
        .addEventListener("click", refreshLogLevel);
    document.getElementById("refreshMetricsBtn")
        .addEventListener("click", refreshMetrics);

    // Search filters for new tabs
    document.getElementById("envSearchInput")
        .addEventListener("input", filterEnv);
    document.getElementById("loggerSearchInput")
        .addEventListener("input", filterLoggers);
    document.getElementById("metricsSearchInput")
        .addEventListener("input", filterMetrics);

    // Initial load
    loadVersionInfo();
    refreshAll();
});

function updateCurrentTime() {
    var now = new Date();
    var timeStr = now.toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
    document.getElementById("currentTime").textContent = timeStr;
}

function toggleAutoRefresh() {
    var toggle = document.getElementById("autoRefreshToggle");
    if (toggle.checked) {
        autoRefreshTimer = setInterval(refreshAll, 5000);
    } else {
        if (autoRefreshTimer) {
            clearInterval(autoRefreshTimer);
            autoRefreshTimer = null;
        }
    }
}

async function loadVersionInfo() {
    try {
        var results = await Promise.all([
            fetch(baseUrl + "/version"),
            fetch(baseUrl + "/commit_id")
        ]);
        var version = results[0].ok
            ? await results[0].json() : "unknown";
        var commitId = results[1].ok
            ? await results[1].json() : "unknown";
        var shortCommit = typeof commitId === "string"
            ? commitId.substring(0, 8) : commitId;
        var badge = document.getElementById("versionBadge");
        badge.textContent = "v" + version
            + " (" + shortCommit + ")";
        badge.style.display = "inline";
    } catch (err) {
        console.warn("Failed to load version info:", err);
    }
}

async function refreshAll() {
    var statusEl = document.getElementById("connectionStatus");
    try {
        statusEl.textContent = "Refreshing...";
        statusEl.className = "badge bg-warning";

        // Fetch both endpoints in parallel
        var results = await Promise.all([
            fetch(baseUrl + "/api/status"),
            fetch(baseUrl + "/api/healthcheck")
        ]);

        var statusResp = results[0];
        var healthResp = results[1];

        if (!statusResp.ok) {
            throw new Error(
                "Status API returned " + statusResp.status
            );
        }

        statusData = await statusResp.json();
        var healthData = healthResp.ok
            ? await healthResp.json()
            : { status: "unknown" };

        statusData._health = healthData;

        statusEl.textContent = "Connected";
        statusEl.className = "badge bg-success";

        renderAll(statusData);
    } catch (err) {
        console.error("Refresh error:", err);
        statusEl.textContent = "Error: " + err.message;
        statusEl.className = "badge bg-danger";
    }
}

function renderAll(data) {
    renderOverview(data);
    renderStorage(data);
    renderGpuContexts(data);
    renderRawJson(data);
    // Refresh all auxiliary tabs
    refreshPeriodicThreads();
    refreshThreads();
    refreshEnv();
    refreshLogLevel();
    refreshMetrics();
}

// ---------------------------------------------------------------
// Overview Tab
// ---------------------------------------------------------------
function renderOverview(data) {
    var container = document.getElementById("overviewCards");
    var isHealthy = data.is_healthy;
    var healthClass = isHealthy ? "healthy" : "unhealthy";
    var healthText = isHealthy ? "Healthy" : "Unhealthy";

    var sm = data.storage_manager || {};
    var l1 = sm.l1_manager || {};
    var l1TotalBytes = l1.memory_total_bytes || 0;
    var l1UsedBytes = l1.memory_used_bytes || 0;
    var l1Pct = l1TotalBytes > 0
        ? Math.round((l1UsedBytes / l1TotalBytes) * 100)
        : 0;
    var l1Objects = l1.total_object_count || 0;

    var barColor = l1Pct > 90
        ? "#dc3545"
        : l1Pct > 70
            ? "#ffc107"
            : "#198754";

    var gpuIds = data.registered_gpu_ids || [];
    var sessions = data.active_sessions || 0;
    var engineType = data.engine_type || "Unknown";
    var chunkSize = data.chunk_size || "N/A";
    var hashAlgo = data.hash_algorithm || "N/A";
    var numAdapters = sm.num_l2_adapters || 0;

    var html = "";

    // Row 1: Health + Engine Info
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">Health</div>';
    html += '      <div class="mt-2">';
    html += '        <span class="health-dot ' + healthClass;
    html += '"></span>';
    html += '        <span class="fs-4 fw-bold">';
    html += healthText + "</span>";
    html += "      </div>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">Engine Type</div>';
    html += '      <div class="stat-value fs-4">';
    html += engineType + "</div>";
    html += '      <small class="text-muted">Chunk: ';
    html += chunkSize + " | Hash: " + hashAlgo + "</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">Active Sessions</div>';
    html += '      <div class="stat-value">' + sessions + "</div>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    // Row 2: GPU + L1 + L2
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">GPU Workers</div>';
    html += '      <div class="stat-value">';
    html += gpuIds.length + "</div>";
    html += '      <small class="text-muted">IDs: ';
    html += (gpuIds.length > 0
        ? gpuIds.join(", ")
        : "none") + "</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">L1 Cache Usage</div>';
    html += '      <div class="memory-bar mt-2">';
    html += '        <div class="bar-fill" style="width:';
    html += l1Pct + "%;background-color:" + barColor + '">';
    html += l1Pct + "%</div>";
    html += "      </div>";
    html += '      <small class="text-muted mt-1 d-block">';
    html += formatBytes(l1UsedBytes) + " / ";
    html += formatBytes(l1TotalBytes);
    html += " (" + l1Objects + " objects)</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">L2 Adapters</div>';
    html += '      <div class="stat-value">' + numAdapters + "</div>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    // Row 3: Prefetch & Pending
    var pendingLookups = data.pending_lookup_count || 0;
    var nextJobId = data.next_prefetch_job_id || 0;

    html += '<div class="col-12 mt-2 mb-2">';
    html += '  <h5 class="text-muted">';
    html += '    <i class="bi bi-hourglass-split"></i>';
    html += "    Pending & Prefetch";
    html += "  </h5>";
    html += "</div>";

    // Prefetch jobs: count + ID list in one card
    var prefetchJobIds = data.prefetch_job_ids || [];
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">';
    html += "Active Prefetch Jobs</div>";
    html += '      <div class="stat-value">';
    html += prefetchJobIds.length + "</div>";
    html += '      <small class="text-muted">';
    html += "next ID: " + nextJobId;
    if (prefetchJobIds.length > 0) {
        html += " &middot; IDs: ";
        html += escapeHtml(
            prefetchJobIds.slice(0, 5).join(", ")
        );
        if (prefetchJobIds.length > 5) {
            html += " +" + (prefetchJobIds.length - 5)
                + " more";
        }
    }
    html += "</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">';
    html += "Pending Lookups</div>";
    html += '      <div class="stat-value">';
    html += pendingLookups + "</div>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    // Pending request IDs (collapsed list)
    var pendingReqIds = data.pending_request_ids || [];
    var pendingLookupIds = data.pending_lookup_request_ids || [];
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">';
    html += "Pending Requests</div>";
    html += '      <div class="stat-value">';
    html += pendingReqIds.length + "</div>";
    if (pendingReqIds.length > 0) {
        html += '      <small class="text-muted">';
        html += escapeHtml(
            pendingReqIds.slice(0, 3).join(", ")
        );
        if (pendingReqIds.length > 3) {
            html += " +" + (pendingReqIds.length - 3)
                + " more";
        }
        html += "</small>";
    }
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    // Row 4: Periodic Threads summary
    var pt = data.periodic_threads || {};
    var ptTotal = pt.total_count || 0;
    var ptRunning = pt.running_count || 0;
    var ptActive = pt.active_count || 0;

    if (ptTotal > 0) {
        html += '<div class="col-12 mt-2 mb-2">';
        html += '  <h5 class="text-muted">';
        html += '    <i class="bi bi-arrow-repeat"></i>';
        html += "    Periodic Threads";
        html += "  </h5>";
        html += "</div>";

        html += '<div class="col-md-4 mb-3">';
        html += '  <div class="card stat-card">';
        html += '    <div class="card-body">';
        html += '      <div class="stat-label">';
        html += "Registered</div>";
        html += '      <div class="stat-value">';
        html += ptTotal + "</div>";
        html += "    </div>";
        html += "  </div>";
        html += "</div>";

        html += '<div class="col-md-4 mb-3">';
        html += '  <div class="card stat-card">';
        html += '    <div class="card-body">';
        html += '      <div class="stat-label">';
        html += "Running</div>";
        var runColor = ptRunning === ptTotal
            ? "#198754" : "#ffc107";
        html += '      <div class="stat-value" style="color:';
        html += runColor + '">';
        html += ptRunning + " / " + ptTotal + "</div>";
        html += "    </div>";
        html += "  </div>";
        html += "</div>";

        html += '<div class="col-md-4 mb-3">';
        html += '  <div class="card stat-card">';
        html += '    <div class="card-body">';
        html += '      <div class="stat-label">';
        html += "Active</div>";
        var actColor = ptActive === ptRunning
            ? "#198754" : "#dc3545";
        html += '      <div class="stat-value" style="color:';
        html += actColor + '">';
        html += ptActive + " / " + ptRunning + "</div>";
        html += "    </div>";
        html += "  </div>";
        html += "</div>";
    }

    // Row 5: Hit Statistics
    html += renderHitStats(data.hit_stats);

    container.innerHTML = html;
}

// ---------------------------------------------------------------
// Hit Statistics (Overview sub-section)
// ---------------------------------------------------------------
function renderHitStats(stats) {
    if (!stats) {
        return "";
    }

    var hitRate = stats.hit_rate || 0;
    var hitPct = Math.round(hitRate * 100);
    var hitColor = hitPct >= 80
        ? "#198754"
        : hitPct >= 50
            ? "#ffc107"
            : "#dc3545";

    var totalReqs = stats.total_requests || 0;
    var totalTokens = stats.total_tokens || 0;
    var retrievedTokens = stats.total_retrieved_tokens || 0;

    var html = "";

    // Section divider
    html += '<div class="col-12 mt-2 mb-2">';
    html += '  <h5 class="text-muted">';
    html += '    <i class="bi bi-bullseye"></i>';
    html += "    Hit Statistics";
    html += "  </h5>";
    html += "</div>";

    // Hit rate card
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">GPU Hit Rate</div>';
    html += '      <div class="stat-value" style="color:';
    html += hitColor + '">' + hitPct + "%</div>";
    html += '      <div class="memory-bar mt-2">';
    html += '        <div class="bar-fill" style="width:';
    html += hitPct + "%;background-color:";
    html += hitColor + '">' + hitPct + "%</div>";
    html += "      </div>";
    html += '      <small class="text-muted mt-1 d-block">';
    html += formatTokenCount(retrievedTokens);
    html += " / " + formatTokenCount(totalTokens);
    html += " tokens</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    // Total requests card
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">Total Requests</div>';
    html += '      <div class="stat-value">';
    html += totalReqs + "</div>";
    html += '      <small class="text-muted">';
    html += formatTokenCount(totalTokens);
    html += " tokens total</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    // Retrieved tokens card
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">';
    html += "GPU Retrieved</div>";
    html += '      <div class="stat-value">';
    html += formatTokenCount(retrievedTokens) + "</div>";
    html += '      <small class="text-muted">';
    html += "tokens written to GPU</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    return html;
}

function formatTokenCount(count) {
    if (count >= 1000000) {
        return (count / 1000000).toFixed(1) + "M";
    }
    if (count >= 1000) {
        return (count / 1000).toFixed(1) + "K";
    }
    return String(count);
}

function formatBytes(bytes) {
    if (bytes >= 1073741824) {
        return (bytes / 1073741824).toFixed(2) + " GB";
    }
    if (bytes >= 1048576) {
        return (bytes / 1048576).toFixed(1) + " MB";
    }
    if (bytes >= 1024) {
        return (bytes / 1024).toFixed(1) + " KB";
    }
    return bytes + " B";
}

// ---------------------------------------------------------------
// Storage Tab
// ---------------------------------------------------------------
function renderStorage(data) {
    var container = document.getElementById("storageContent");
    var sm = data.storage_manager;
    if (!sm) {
        container.innerHTML = '<div class="alert alert-warning">'
            + "No storage manager data available</div>";
        return;
    }

    var html = "";

    // L1 Manager
    html += renderSection(
        "L1 Manager", sm.l1_manager, "l1-section"
    );

    // Store Controller
    html += renderSection(
        "Store Controller",
        sm.store_controller,
        "store-section"
    );

    // Prefetch Controller
    html += renderSection(
        "Prefetch Controller",
        sm.prefetch_controller,
        "prefetch-section"
    );

    // L1 Eviction Controller
    html += renderSection(
        "L1 Eviction Controller",
        sm.l1_eviction_controller,
        "l1-evict-section"
    );

    // L2 Eviction Controller
    html += renderSection(
        "L2 Eviction Controller",
        sm.l2_eviction_controller,
        "l2-evict-section"
    );

    // L2 Adapters
    var adapters = sm.l2_adapters || [];
    for (var i = 0; i < adapters.length; i++) {
        html += renderSection(
            "L2 Adapter #" + i,
            adapters[i],
            "l2-adapter-" + i
        );
    }

    container.innerHTML = html;

    // Attach toggle listeners
    container.querySelectorAll(".section-header")
        .forEach(function(header) {
            header.addEventListener("click", function() {
                var targetId = this.dataset.target;
                var body = document.getElementById(targetId);
                if (body) {
                    body.classList.toggle("d-none");
                    this.classList.toggle("collapsed");
                }
            });
        });
}

function renderSection(title, obj, sectionId) {
    if (!obj) {
        return "";
    }
    var isHealthy = obj.is_healthy;
    var dotClass = isHealthy === true
        ? "healthy"
        : isHealthy === false
            ? "unhealthy"
            : "";

    var html = "";
    html += '<div class="section-header" data-target="';
    html += sectionId + '-body">';
    html += "  <span>";
    if (dotClass) {
        html += '<span class="health-dot ' + dotClass;
        html += '"></span>';
    }
    html += "    <strong>" + title + "</strong>";
    html += "  </span>";
    html += '  <i class="bi bi-chevron-down toggle-icon"></i>';
    html += "</div>";
    html += '<div id="' + sectionId + '-body" class="mb-3">';
    html += '  <div class="card"><div class="card-body">';
    html += renderObjectTable(obj);
    html += "  </div></div>";
    html += "</div>";
    return html;
}

function renderObjectTable(obj) {
    if (obj === null || obj === undefined) {
        return '<span class="text-muted">N/A</span>';
    }
    if (typeof obj !== "object") {
        return "<span>" + escapeHtml(String(obj)) + "</span>";
    }
    if (Array.isArray(obj)) {
        if (obj.length === 0) {
            return '<span class="text-muted">[]</span>';
        }
        var html = '<ul class="list-group list-group-flush">';
        for (var i = 0; i < obj.length; i++) {
            html += "<li class=\"list-group-item\">";
            html += renderObjectTable(obj[i]);
            html += "</li>";
        }
        html += "</ul>";
        return html;
    }

    var keys = Object.keys(obj);
    if (keys.length === 0) {
        return '<span class="text-muted">{}</span>';
    }

    var html = '<table class="table table-sm table-bordered '
        + 'mb-0"><tbody>';
    for (var k = 0; k < keys.length; k++) {
        var key = keys[k];
        var val = obj[key];
        html += "<tr>";
        html += '<td class="fw-bold" style="width:30%">';
        html += escapeHtml(key) + "</td>";
        html += "<td>";
        if (typeof val === "object" && val !== null) {
            html += renderObjectTable(val);
        } else if (typeof val === "boolean") {
            html += val
                ? '<span class="badge bg-success">true</span>'
                : '<span class="badge bg-danger">false</span>';
        } else {
            html += escapeHtml(String(val));
        }
        html += "</td></tr>";
    }
    html += "</tbody></table>";
    return html;
}

// ---------------------------------------------------------------
// GPU Contexts Tab
// ---------------------------------------------------------------
function renderGpuContexts(data) {
    var container = document.getElementById("gpuContent");
    var meta = data.gpu_context_meta;
    if (!meta || Object.keys(meta).length === 0) {
        container.innerHTML = '<div class="alert alert-info">'
            + "No GPU contexts registered yet. "
            + "Workers will register when they connect."
            + "</div>";
        return;
    }

    var html = "";
    var gpuIds = Object.keys(meta);
    for (var i = 0; i < gpuIds.length; i++) {
        var gpuId = gpuIds[i];
        var ctx = meta[gpuId];
        var layout = ctx.kv_cache_layout || {};

        html += '<div class="card mb-3">';
        html += '  <div class="card-header bg-light">';
        html += '    <i class="bi bi-gpu-card"></i> ';
        html += "    <strong>GPU Worker " + gpuId + "</strong>";
        html += '    <span class="badge bg-primary ms-2">';
        html += (ctx.model_name || "unknown") + "</span>";
        html += "  </div>";
        html += '  <div class="card-body">';

        // Basic info
        html += '  <div class="row mb-3">';
        html += '    <div class="col-md-6">';
        html += '      <table class="table table-sm mb-0">';
        html += "        <tbody>";
        html += "          <tr><td class=\"fw-bold\">";
        html += "Model</td><td>";
        html += escapeHtml(ctx.model_name || "N/A");
        html += "</td></tr>";
        html += "          <tr><td class=\"fw-bold\">";
        html += "World Size</td><td>";
        html += ctx.world_size + "</td></tr>";
        html += "        </tbody></table>";
        html += "    </div>";
        html += '    <div class="col-md-6">';

        if (Object.keys(layout).length > 0) {
            html += '<table class="table table-sm mb-0">';
            html += "  <tbody>";
            var layoutKeys = Object.keys(layout);
            for (var j = 0; j < layoutKeys.length; j++) {
                var lk = layoutKeys[j];
                var lv = layout[lk];
                html += "  <tr><td class=\"fw-bold\">";
                html += escapeHtml(lk) + "</td><td>";
                if (typeof lv === "object" && lv !== null) {
                    html += "<code>";
                    html += escapeHtml(JSON.stringify(lv));
                    html += "</code>";
                } else if (typeof lv === "boolean") {
                    html += lv
                        ? '<span class="badge bg-success">'
                          + "true</span>"
                        : '<span class="badge bg-secondary">'
                          + "false</span>";
                } else {
                    html += escapeHtml(String(lv));
                }
                html += "</td></tr>";
            }
            html += "  </tbody></table>";
        } else {
            html += '<span class="text-muted">';
            html += "No layout info</span>";
        }

        html += "    </div>";
        html += "  </div>";
        html += "  </div>";
        html += "</div>";
    }

    container.innerHTML = html;
}

// ---------------------------------------------------------------
// Raw JSON Tab
// ---------------------------------------------------------------
function renderRawJson(data) {
    var el = document.getElementById("rawJsonContent");
    el.textContent = JSON.stringify(data, null, 2);
}

function filterJson() {
    var input = document.getElementById("jsonSearchInput");
    var el = document.getElementById("rawJsonContent");
    var term = input.value.toLowerCase();

    if (!statusData) {
        return;
    }

    var fullText = JSON.stringify(statusData, null, 2);
    if (!term) {
        el.textContent = fullText;
        return;
    }

    var lines = fullText.split("\n");
    var filtered = lines.filter(function(line) {
        return line.toLowerCase().indexOf(term) !== -1;
    });
    el.textContent = filtered.join("\n") || "No matches found";
}

// ---------------------------------------------------------------
// Periodic Threads Tab
// ---------------------------------------------------------------
var periodicThreadsData = null;

async function refreshPeriodicThreads() {
    var el = document.getElementById("periodicThreadsContent");
    try {
var resp = await fetch(
            baseUrl + "/periodic-threads"
        );
        if (!resp.ok) {
            throw new Error("HTTP " + resp.status);
        }
        periodicThreadsData = await resp.json();
        renderPeriodicThreads(periodicThreadsData);
    } catch (err) {
        el.innerHTML = '<div class="alert alert-danger">'
            + "Failed to load: " + escapeHtml(err.message)
            + "</div>";
    }
}

function renderPeriodicThreads(data) {
    var el = document.getElementById("periodicThreadsContent");
    var summary = data.summary || {};
    var threads = data.threads || [];

    var html = "";

    // Summary cards
    html += '<div class="row mb-3">';
    html += renderMiniCard(
        "Total", summary.total_count || 0, "bi-layers"
    );
    html += renderMiniCard(
        "Running", summary.running_count || 0,
        "bi-play-circle", "text-success"
    );
    html += renderMiniCard(
        "Active", summary.active_count || 0,
        "bi-activity", "text-primary"
    );
    html += "</div>";

    // By level breakdown
    var byLevel = summary.by_level || {};
    var levelKeys = Object.keys(byLevel);
    if (levelKeys.length > 0) {
        html += '<div class="row mb-3">';
        for (var i = 0; i < levelKeys.length; i++) {
            var lk = levelKeys[i];
            var lv = byLevel[lk];
            html += '<div class="col-md-3 mb-2">';
            html += '  <div class="card">';
            html += '    <div class="card-body p-2 text-center">';
            html += '      <small class="text-muted">';
            html += escapeHtml(lk.toUpperCase());
            html += "</small><br>";
            html += '      <span class="fw-bold">';
            html += (lv.running || 0) + "/" + (lv.total || 0);
            html += "</span> running";
            html += "    </div></div></div>";
        }
        html += "</div>";
    }

    // Thread table
    if (threads.length > 0) {
        html += '<table class="table table-sm table-hover">';
        html += "<thead><tr>";
        html += '<th>Name</th><th>Level</th>';
        html += '<th>Status</th><th>Interval</th>';
        html += '<th>Runs</th><th>Failures</th>';
        html += '<th>Last Run</th><th>Last Summary</th>';
        html += "</tr></thead><tbody>";
        for (var j = 0; j < threads.length; j++) {
            var t = threads[j];
            var statusBadge = t.is_running
                ? (t.is_active
                    ? '<span class="badge bg-success">' +
                      'Active</span>'
                    : '<span class="badge bg-warning">' +
                      'Stale</span>')
                : '<span class="badge bg-secondary">' +
                  'Stopped</span>';
            html += "<tr>";
            html += "<td><code>" + escapeHtml(t.name || "")
                + "</code></td>";
            html += "<td>" + escapeHtml(t.level || "")
                + "</td>";
            html += "<td>" + statusBadge + "</td>";
            html += "<td>" + (t.interval || "N/A")
                + "s</td>";
            html += "<td>" + (t.total_runs || 0) + "</td>";
            html += "<td>" + (t.total_failures || 0)
                + "</td>";
            html += "<td>" + escapeHtml(
                t.last_run_ago || "never"
            ) + "</td>";
            if (t.last_summary) {
                var s = t.last_summary;
                var pills = "";
                if (s.success !== undefined) {
                    pills += s.success
                        ? '<span class="badge bg-success me-1">'
                          + "OK</span>"
                        : '<span class="badge bg-danger me-1">'
                          + "FAIL</span>";
                }
                if (s.duration_ms !== undefined) {
                    pills += '<span class="badge bg-info '
                        + 'text-dark me-1">'
                        + s.duration_ms.toFixed(1)
                        + "ms</span>";
                }
                if (s.message) {
                    pills += "<span>"
                        + escapeHtml(s.message) + "</span>";
                }
                var fullJson = JSON.stringify(
                    s, null, 2
                );
                var uid = "summary-" + j;
                html += "<td>" + pills
                    + ' <a href="#" class="small" '
                    + 'data-bs-toggle="collapse" '
                    + 'data-bs-target="#' + uid + '">'
                    + "detail</a>"
                    + '<div class="collapse" id="' + uid
                    + '"><pre class="mb-0 mt-1" '
                    + 'style="font-size:.75rem;'
                    + 'max-height:200px;overflow:auto">'
                    + escapeHtml(fullJson)
                    + "</pre></div></td>";
            } else {
                html += "<td></td>";
            }
            html += "</tr>";
        }
        html += "</tbody></table>";
    } else {
        html += '<div class="alert alert-info">';
        html += "No periodic threads registered.";
        html += "</div>";
    }

    el.innerHTML = html;
}

function renderMiniCard(label, value, icon, colorClass) {
    var cls = colorClass || "";
    var html = '<div class="col-md-2 mb-2">';
    html += '  <div class="card text-center">';
    html += '    <div class="card-body p-2">';
    html += '      <i class="bi ' + icon + ' ' + cls;
    html += '"></i><br>';
    html += '      <span class="fs-5 fw-bold ' + cls + '">';
    html += value + "</span><br>";
    html += '      <small class="text-muted">';
    html += escapeHtml(label) + "</small>";
    html += "    </div></div></div>";
    return html;
}

// ---------------------------------------------------------------
// Threads Tab
// ---------------------------------------------------------------
async function refreshThreads() {
    var el = document.getElementById("threadsContent");
    try {
        el.textContent = "Loading...";
var resp = await fetch(baseUrl + "/threads");
        if (!resp.ok) {
            throw new Error("HTTP " + resp.status);
        }
        el.textContent = await resp.text();
    } catch (err) {
        el.textContent = "Error: " + err.message;
    }
}

// ---------------------------------------------------------------
// Env Tab
// ---------------------------------------------------------------
var envRawText = "";

async function refreshEnv() {
    var el = document.getElementById("envContent");
    try {
        el.textContent = "Loading...";
var resp = await fetch(baseUrl + "/env");
        if (!resp.ok) {
            throw new Error("HTTP " + resp.status);
        }
        envRawText = await resp.text();
        el.textContent = envRawText;
    } catch (err) {
        el.textContent = "Error: " + err.message;
    }
}

function filterEnv() {
    var term = document.getElementById("envSearchInput")
        .value.toLowerCase();
    var el = document.getElementById("envContent");
    if (!term || !envRawText) {
        el.textContent = envRawText;
        return;
    }
    var lines = envRawText.split("\n");
    var filtered = lines.filter(function(line) {
        return line.toLowerCase().indexOf(term) !== -1;
    });
    el.textContent = filtered.join("\n")
        || "No matches found";
}

// ---------------------------------------------------------------
// Log Level Tab
// ---------------------------------------------------------------
var logLevelRawText = "";

async function refreshLogLevel() {
    var el = document.getElementById("logLevelContent");
    try {
        el.textContent = "Loading...";
var resp = await fetch(baseUrl + "/loglevel");
        if (!resp.ok) {
            throw new Error("HTTP " + resp.status);
        }
        logLevelRawText = await resp.text();
        el.textContent = logLevelRawText;
    } catch (err) {
        el.textContent = "Error: " + err.message;
    }
}

function filterLoggers() {
    var term = document.getElementById("loggerSearchInput")
        .value.toLowerCase();
    var el = document.getElementById("logLevelContent");
    if (!term || !logLevelRawText) {
        el.textContent = logLevelRawText;
        return;
    }
    var lines = logLevelRawText.split("\n");
    var filtered = lines.filter(function(line) {
        return line.toLowerCase().indexOf(term) !== -1;
    });
    el.textContent = filtered.join("\n")
        || "No matches found";
}

// ---------------------------------------------------------------
// Metrics Tab
// ---------------------------------------------------------------
var metricsRawText = "";

async function refreshMetrics() {
    var el = document.getElementById("metricsContent");
    try {
        el.textContent = "Loading...";
var resp = await fetch(baseUrl + "/metrics");
        if (!resp.ok) {
            throw new Error("HTTP " + resp.status);
        }
        metricsRawText = await resp.text();
        el.textContent = metricsRawText;
    } catch (err) {
        el.textContent = "Error: " + err.message;
    }
}

function filterMetrics() {
    var term = document.getElementById("metricsSearchInput")
        .value.toLowerCase();
    var el = document.getElementById("metricsContent");
    if (!term || !metricsRawText) {
        el.textContent = metricsRawText;
        return;
    }
    var lines = metricsRawText.split("\n");
    var filtered = lines.filter(function(line) {
        return line.toLowerCase().indexOf(term) !== -1;
    });
    el.textContent = filtered.join("\n")
        || "No matches found";
}

// ---------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------
function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

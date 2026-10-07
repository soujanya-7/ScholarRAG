/**
 * ScholarRAG Studio — Interactive Frontend Logic (Vanguard Tier)
 */

document.addEventListener("DOMContentLoaded", () => {
    // Application State
    const state = {
        apiKey: localStorage.getItem("scholar_rag_api_key") || "",
        stats: null,
        chunks: [],
        activeTab: "tab-chat",
        retrievalMode: "hybrid_rerank",
        topK: 8,
        topN: 4,
        useHyde: false,
    };

    // DOM Elements
    const statPapers = document.getElementById("stat-papers");
    const statChunks = document.getElementById("stat-chunks");
    const apiStatusDot = document.getElementById("api-status-dot");
    const apiStatusText = document.getElementById("api-status-text");
    const libraryCount = document.getElementById("library-count");
    const paperList = document.getElementById("paper-list");
    
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const btnLoadSample = document.getElementById("btn-load-sample");
    const btnClearCorpus = document.getElementById("btn-clear-corpus");

    const retrievalSelect = document.getElementById("retrieval-mode-select");
    const topKSlider = document.getElementById("top-k-slider");
    const topKVal = document.getElementById("top-k-val");
    const topNSlider = document.getElementById("top-n-slider");
    const topNVal = document.getElementById("top-n-val");
    const hydeToggle = document.getElementById("hyde-toggle");

    const tabBtns = document.querySelectorAll(".tab-pill");
    const tabContents = document.querySelectorAll(".tab-panel");

    // Chat Elements
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const btnSendChat = document.getElementById("btn-send-chat");
    const promptPills = document.querySelectorAll(".prompt-pill");

    // Inspector Elements
    const inspectorQueryInput = document.getElementById("inspector-query-input");
    const btnRunBenchmark = document.getElementById("btn-run-benchmark");
    const denseCards = document.getElementById("dense-cards");
    const bm25Cards = document.getElementById("bm25-cards");
    const hybridCards = document.getElementById("hybrid-cards");
    const rerankCards = document.getElementById("rerank-cards");
    const denseTimer = document.getElementById("dense-timer");
    const bm25Timer = document.getElementById("bm25-timer");
    const hybridTimer = document.getElementById("hybrid-timer");
    const rerankTimer = document.getElementById("rerank-timer");

    // Explorer Elements
    const chunkSearchInput = document.getElementById("chunk-search-input");
    const chunkSourceFilter = document.getElementById("chunk-source-filter");
    const chunksTableBody = document.getElementById("chunks-table-body");

    // Modals
    const settingsModal = document.getElementById("settings-modal");
    const btnOpenSettings = document.getElementById("btn-open-settings");
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const btnCancelSettings = document.getElementById("btn-cancel-settings");
    const btnSaveSettings = document.getElementById("btn-save-settings");
    const geminiKeyInput = document.getElementById("gemini-key-input");

    const excerptModal = document.getElementById("excerpt-modal");
    const btnCloseExcerpt = document.getElementById("btn-close-excerpt");
    const excerptModalTitle = document.getElementById("excerpt-modal-title");
    const excerptModalMeta = document.getElementById("excerpt-modal-meta");
    const excerptModalContent = document.getElementById("excerpt-modal-content");

    // ==========================================
    // INITIALIZATION & STATS
    // ==========================================
    async function fetchStats() {
        try {
            const res = await fetch("/api/stats");
            const data = await res.json();
            state.stats = data;

            if (statPapers) statPapers.textContent = data.total_documents;
            if (statChunks) statChunks.textContent = data.total_chunks;
            if (libraryCount) libraryCount.textContent = `${data.total_documents} Docs`;

            // Check API key status
            const hasKey = data.has_api_key || !!state.apiKey;
            if (hasKey) {
                apiStatusDot.classList.add("active");
                apiStatusText.textContent = "API Ready";
            } else {
                apiStatusDot.classList.remove("active");
                apiStatusText.textContent = "Set API Key";
            }

            renderLibraryList(data.documents);
        } catch (err) {
            console.error("Failed to fetch stats:", err);
        }
    }

    function renderLibraryList(docs) {
        if (!docs || docs.length === 0) {
            paperList.innerHTML = `<div class="empty-state-micro">No documents indexed yet</div>`;
            return;
        }

        paperList.innerHTML = docs.map(doc => `
            <div class="paper-row">
                <span class="paper-name" title="${doc.source}">📄 ${doc.source}</span>
                <span class="paper-stats-mono">${doc.total_chunks}c · ${doc.total_pages}p</span>
            </div>
        `).join("");

        // Populate chunk filter
        const sources = ["all", ...docs.map(d => d.source)];
        if (chunkSourceFilter) {
            chunkSourceFilter.innerHTML = sources.map(s => `
                <option value="${s}">${s === "all" ? "All Documents" : s}</option>
            `).join("");
        }
    }

    // ==========================================
    // TABS SWITCHING
    // ==========================================
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(tabId).classList.add("active");
            state.activeTab = tabId;

            if (tabId === "tab-chunks") {
                loadAllChunks();
            }
        });
    });

    // ==========================================
    // HYPERPARAMETERS
    // ==========================================
    if (retrievalSelect) {
        retrievalSelect.addEventListener("change", (e) => {
            state.retrievalMode = e.target.value;
            const tag = document.getElementById("pipeline-active-tag");
            if (tag) tag.textContent = `Strategy: ${e.target.options[e.target.selectedIndex].text.split("(")[0].trim()}`;
        });
    }

    if (topKSlider) {
        topKSlider.addEventListener("input", (e) => {
            state.topK = parseInt(e.target.value);
            topKVal.textContent = state.topK;
        });
    }

    if (topNSlider) {
        topNSlider.addEventListener("input", (e) => {
            state.topN = parseInt(e.target.value);
            topNVal.textContent = state.topN;
        });
    }

    if (hydeToggle) {
        hydeToggle.addEventListener("change", (e) => {
            state.useHyde = e.target.checked;
        });
    }

    // ==========================================
    // DOCUMENT INGESTION (Upload & Sample)
    // ==========================================
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            dropzone.classList.add("dragover");
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.status === "success") {
                await fetchStats();
                showToast(`Indexed ${file.name} (${data.result.chunks_created} chunks)!`);
            } else {
                alert(`Upload failed: ${data.detail || "Error"}`);
            }
        } catch (err) {
            alert(`Upload error: ${err.message}`);
        } finally {
            dropzone.classList.remove("dragover");
        }
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                Array.from(e.target.files).forEach(uploadFile);
            }
        });
    }

    if (dropzone) {
        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
        dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                Array.from(e.dataTransfer.files).forEach(uploadFile);
            }
        });
    }

    if (btnLoadSample) {
        btnLoadSample.addEventListener("click", async () => {
            btnLoadSample.disabled = true;
            btnLoadSample.innerHTML = `<span class="btn-label-text">Indexing...</span><span class="btn-icon-tray-sm"><i data-lucide="loader" class="spin"></i></span>`;
            try {
                const res = await fetch("/api/load-sample", { method: "POST" });
                const data = await res.json();
                if (data.status === "success") {
                    await fetchStats();
                    showToast("Sample paper 'Attention Is All You Need' indexed!");
                }
            } catch (err) {
                alert("Failed to load sample: " + err.message);
            } finally {
                btnLoadSample.disabled = false;
                btnLoadSample.innerHTML = `<span class="btn-label-text">✨ Load Transformer Paper</span><span class="btn-icon-tray-sm"><i data-lucide="sparkles"></i></span>`;
                lucide.createIcons();
            }
        });
    }

    if (btnClearCorpus) {
        btnClearCorpus.addEventListener("click", async () => {
            if (!confirm("Are you sure you want to reset all indexed documents from ChromaDB and BM25?")) return;
            try {
                await fetch("/api/clear", { method: "POST" });
                await fetchStats();
                if (chunksTableBody) {
                    chunksTableBody.innerHTML = `<tr><td colspan="6" class="text-center-muted">No chunks loaded</td></tr>`;
                }
                showToast("Knowledge base reset!");
            } catch (err) {
                alert("Error clearing corpus: " + err.message);
            }
        });
    }

    // ==========================================
    // GROUNDED CHAT STUDIO
    // ==========================================
    async function sendChatMessage(query) {
        if (!query.trim()) return;

        appendMessage("user", query);
        chatInput.value = "";

        const loadingId = "msg-loading-" + Date.now();
        appendLoadingMessage(loadingId);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    mode: state.retrievalMode,
                    top_k: state.topK,
                    top_n_rerank: state.topN,
                    use_hyde: state.useHyde,
                    api_key: state.apiKey || undefined
                })
            });

            const data = await res.json();
            removeMessage(loadingId);

            if (!res.ok) {
                appendMessage("assistant", `⚠️ **Error:** ${data.detail || "Unable to generate answer. Please check your Gemini API key."}`);
                if (data.detail && data.detail.includes("API key")) {
                    openSettingsModal();
                }
                return;
            }

            appendAssistantResponse(data);

        } catch (err) {
            removeMessage(loadingId);
            appendMessage("assistant", `⚠️ **Network Error:** ${err.message}`);
        }
    }

    function appendMessage(role, text) {
        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = `chat-bubble ${role}-bubble`;
        bubbleDiv.innerHTML = `
            <div class="bubble-avatar">
                <i data-lucide="${role === 'user' ? 'user' : 'sparkles'}"></i>
            </div>
            <div class="bubble-body">
                <div class="bubble-meta">
                    <span class="bubble-author">${role === 'user' ? 'You' : 'ScholarRAG Assistant'}</span>
                </div>
                <div class="bubble-text">${marked.parse(text)}</div>
            </div>
        `;
        chatMessages.appendChild(bubbleDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        lucide.createIcons();
    }

    function appendLoadingMessage(id) {
        const bubbleDiv = document.createElement("div");
        bubbleDiv.id = id;
        bubbleDiv.className = "chat-bubble assistant-bubble";
        bubbleDiv.innerHTML = `
            <div class="bubble-avatar">
                <i data-lucide="sparkles"></i>
            </div>
            <div class="bubble-body">
                <div class="bubble-meta">
                    <span class="bubble-author">ScholarRAG Assistant</span>
                    <span class="badge-grounded">Retrieving & Re-Ranking...</span>
                </div>
                <div class="bubble-text">
                    <p>Searching dense & sparse index, scoring via FlashRank cross-encoder, and formulating citations...</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(bubbleDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        lucide.createIcons();
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendAssistantResponse(data) {
        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "chat-bubble assistant-bubble";

        let citationsHtml = "";
        if (data.sources_cited && data.sources_cited.length > 0) {
            citationsHtml = `
                <div class="citations-deck">
                    <span style="font-size: 0.68rem; color: var(--text-muted); display: block; width: 100%; margin-bottom: 2px;">📍 Verified Citations (Click to inspect source):</span>
                    ${data.sources_cited.map(s => `
                        <button class="citation-pill" data-chunk-id="${s.chunk_id || ''}" data-source="${s.source}" data-page="${s.page_number}">
                            <i data-lucide="bookmark"></i> ${s.source} · Page ${s.page_number}
                        </button>
                    `).join("")}
                </div>
            `;
        }

        const statsHtml = `
            <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 6px; font-family: var(--font-mono);">
                ⚡ Retrieval: ${data.stats.retrieval_time_s}s | LLM: ${data.stats.generation_time_s}s | Chunks: ${data.stats.chunks_used}
            </div>
        `;

        bubbleDiv.innerHTML = `
            <div class="bubble-avatar">
                <i data-lucide="sparkles"></i>
            </div>
            <div class="bubble-body">
                <div class="bubble-meta">
                    <span class="bubble-author">ScholarRAG Assistant</span>
                    <span class="badge-grounded">Grounded AI Model</span>
                </div>
                <div class="bubble-text">${marked.parse(data.answer)}</div>
                ${citationsHtml}
                ${statsHtml}
            </div>
        `;

        chatMessages.appendChild(bubbleDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        lucide.createIcons();

        bubbleDiv.querySelectorAll(".citation-pill").forEach(chip => {
            chip.addEventListener("click", () => {
                const chunkId = chip.dataset.chunkId;
                const source = chip.dataset.source;
                const page = chip.dataset.page;

                const chunk = (data.retrieved_chunks || []).find(c => c.chunk_id === chunkId) || {
                    source: source,
                    page_number: page,
                    content: "Passage context retrieved for citation."
                };
                openExcerptModal(chunk);
            });
        });
    }

    if (btnSendChat) {
        btnSendChat.addEventListener("click", () => sendChatMessage(chatInput.value));
    }
    if (chatInput) {
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage(chatInput.value);
            }
        });
    }

    promptPills.forEach(pill => {
        pill.addEventListener("click", () => {
            const q = pill.dataset.query;
            chatInput.value = q;
            sendChatMessage(q);
        });
    });

    // ==========================================
    // MULTI-STRATEGY BENCHMARK INSPECTOR
    // ==========================================
    if (btnRunBenchmark) btnRunBenchmark.addEventListener("click", runBenchmark);
    if (inspectorQueryInput) {
        inspectorQueryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") runBenchmark();
        });
    }

    async function runBenchmark() {
        const query = inspectorQueryInput.value.trim();
        if (!query) return;

        btnRunBenchmark.disabled = true;
        btnRunBenchmark.innerHTML = `<span>Scoring...</span><span class="btn-icon-tray-sm"><i data-lucide="loader" class="spin"></i></span>`;

        [denseCards, bm25Cards, hybridCards, rerankCards].forEach(c => {
            c.innerHTML = `<div class="empty-state-card">Computing relevance...</div>`;
        });

        try {
            const res = await fetch("/api/benchmark-search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    top_k: state.topK,
                    top_n_rerank: state.topN,
                    use_hyde: state.useHyde
                })
            });

            const data = await res.json();
            if (data.status === "success") {
                const s = data.strategies;

                denseTimer.textContent = `${(s.dense.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(denseCards, s.dense.final_chunks, "dense");

                bm25Timer.textContent = `${(s.bm25.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(bm25Cards, s.bm25.final_chunks, "bm25");

                hybridTimer.textContent = `${(s.hybrid.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(hybridCards, s.hybrid.final_chunks, "hybrid");

                rerankTimer.textContent = `${(s.hybrid_rerank.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(rerankCards, s.hybrid_rerank.final_chunks, "rerank");
            }
        } catch (err) {
            alert("Benchmark failed: " + err.message);
        } finally {
            btnRunBenchmark.disabled = false;
            btnRunBenchmark.innerHTML = `<span>Run Benchmark</span><span class="btn-icon-tray-sm"><i data-lucide="play"></i></span>`;
            lucide.createIcons();
        }
    }

    function renderStrategyCards(container, chunks, type) {
        if (!chunks || chunks.length === 0) {
            container.innerHTML = `<div class="empty-state-card">No matching passages found</div>`;
            return;
        }

        container.innerHTML = chunks.map((c, idx) => {
            const rank = idx + 1;
            let scoreDisplay = "";
            let chipClass = `rank-chip rank-chip-${type}`;

            if (type === "dense") scoreDisplay = `Cosine: ${c.score.toFixed(4)}`;
            else if (type === "bm25") scoreDisplay = `BM25: ${c.score.toFixed(2)}`;
            else if (type === "hybrid") scoreDisplay = `RRF: ${c.score.toFixed(6)}`;
            else if (type === "rerank") scoreDisplay = `Re-Rank: ${c.score.toFixed(4)}`;

            return `
                <div class="inspect-card" onclick="window.viewChunk('${encodeURIComponent(JSON.stringify(c))}')">
                    <div class="card-top">
                        <span class="${chipClass}">Rank #${rank}</span>
                        <span class="score-mono">${scoreDisplay}</span>
                    </div>
                    <div class="card-source-tag">
                        <span>📄 ${c.source} · Page ${c.page_number}</span>
                    </div>
                    <div class="card-body-snippet">
                        ${c.content.substring(0, 140)}...
                    </div>
                </div>
            `;
        }).join("");
    }

    window.viewChunk = function(encoded) {
        try {
            const chunk = JSON.parse(decodeURIComponent(encoded));
            openExcerptModal(chunk);
        } catch (e) {
            console.error(e);
        }
    };

    // ==========================================
    // CHUNK VECTOR EXPLORER
    // ==========================================
    async function loadAllChunks() {
        try {
            const res = await fetch("/api/chunks");
            const data = await res.json();
            state.chunks = data.chunks || [];
            renderChunksTable();
        } catch (err) {
            console.error(err);
        }
    }

    function renderChunksTable() {
        if (!chunksTableBody) return;
        const query = chunkSearchInput ? chunkSearchInput.value.toLowerCase() : "";
        const sourceFilter = chunkSourceFilter ? chunkSourceFilter.value : "all";

        const filtered = state.chunks.filter(c => {
            const matchQuery = !query || c.content.toLowerCase().includes(query) || c.chunk_id.toLowerCase().includes(query);
            const matchSource = sourceFilter === "all" || c.source === sourceFilter;
            return matchQuery && matchSource;
        });

        if (filtered.length === 0) {
            chunksTableBody.innerHTML = `<tr><td colspan="6" class="text-center-muted">No matching chunks found</td></tr>`;
            return;
        }

        chunksTableBody.innerHTML = filtered.map(c => `
            <tr>
                <td style="font-family: var(--font-mono);">${c.chunk_index + 1}</td>
                <td style="font-family: var(--font-mono); color: var(--accent-cyan); font-size: 0.72rem;">${c.chunk_id}</td>
                <td>${c.source}</td>
                <td>Page ${c.page_number}</td>
                <td>${c.char_count}</td>
                <td style="cursor: pointer;" onclick="window.viewChunk('${encodeURIComponent(JSON.stringify(c))}')">
                    ${c.content.substring(0, 100)}...
                </td>
            </tr>
        `).join("");
    }

    if (chunkSearchInput) chunkSearchInput.addEventListener("input", renderChunksTable);
    if (chunkSourceFilter) chunkSourceFilter.addEventListener("change", renderChunksTable);

    // ==========================================
    // MODALS
    // ==========================================
    function openSettingsModal() {
        if (geminiKeyInput) geminiKeyInput.value = state.apiKey;
        if (settingsModal) settingsModal.classList.add("open");
    }
    function closeSettingsModal() {
        if (settingsModal) settingsModal.classList.remove("open");
    }

    if (btnOpenSettings) btnOpenSettings.addEventListener("click", openSettingsModal);
    if (btnCloseSettings) btnCloseSettings.addEventListener("click", closeSettingsModal);
    if (btnCancelSettings) btnCancelSettings.addEventListener("click", closeSettingsModal);

    if (btnSaveSettings) {
        btnSaveSettings.addEventListener("click", async () => {
            const key = geminiKeyInput.value.trim();
            state.apiKey = key;
            localStorage.setItem("scholar_rag_api_key", key);

            await fetch("/api/update-config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ gemini_api_key: key })
            });

            closeSettingsModal();
            await fetchStats();
            showToast("Configuration saved successfully!");
        });
    }

    function openExcerptModal(chunk) {
        if (excerptModalTitle) {
            excerptModalTitle.innerHTML = `<i data-lucide="file-text" class="modal-icon"></i> <h3>${chunk.source} (Page ${chunk.page_number})</h3>`;
        }
        if (excerptModalMeta) {
            excerptModalMeta.innerHTML = `Chunk ID: <b>${chunk.chunk_id || 'N/A'}</b> | Length: <b>${chunk.content.length} chars</b> | Score: <b>${chunk.score !== undefined ? chunk.score : 'N/A'}</b>`;
        }
        if (excerptModalContent) {
            excerptModalContent.textContent = chunk.content;
        }
        if (excerptModal) excerptModal.classList.add("open");
        lucide.createIcons();
    }

    function closeExcerptModal() {
        if (excerptModal) excerptModal.classList.remove("open");
    }
    if (btnCloseExcerpt) btnCloseExcerpt.addEventListener("click", closeExcerptModal);

    [settingsModal, excerptModal].forEach(modal => {
        if (modal) {
            modal.addEventListener("click", (e) => {
                if (e.target === modal) modal.classList.remove("open");
            });
        }
    });

    // Toast Alert
    function showToast(msg) {
        const toast = document.createElement("div");
        toast.style.position = "fixed";
        toast.style.bottom = "24px";
        toast.style.right = "24px";
        toast.style.background = "linear-gradient(135deg, rgba(6, 182, 212, 0.9), rgba(99, 102, 241, 0.9))";
        toast.style.backdropFilter = "blur(12px)";
        toast.style.color = "#FFF";
        toast.style.padding = "10px 20px";
        toast.style.borderRadius = "var(--radius-pill)";
        toast.style.border = "1px solid rgba(255, 255, 255, 0.2)";
        toast.style.boxShadow = "0 8px 32px rgba(0,0,0,0.5)";
        toast.style.zIndex = "9999";
        toast.style.fontSize = "0.82rem";
        toast.style.fontWeight = "600";
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3500);
    }

    // Initialize
    fetchStats();
});

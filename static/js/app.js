/**
 * ScholarRAG Studio Interactive Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
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

    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // Chat Elements
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const btnSendChat = document.getElementById("btn-send-chat");
    const promptChips = document.querySelectorAll(".prompt-chip");

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

            statPapers.textContent = data.total_documents;
            statChunks.textContent = data.total_chunks;
            libraryCount.textContent = `${data.total_documents} Docs`;

            // Check API key
            const hasKey = data.has_api_key || !!state.apiKey;
            if (hasKey) {
                apiStatusDot.classList.add("active");
                apiStatusText.textContent = "API Ready";
            } else {
                apiStatusDot.classList.remove("active");
                apiStatusText.textContent = "Set API Key";
            }

            // Populate Library List
            renderLibraryList(data.documents);
        } catch (err) {
            console.error("Failed to fetch stats:", err);
        }
    }

    function renderLibraryList(docs) {
        if (!docs || docs.length === 0) {
            paperList.innerHTML = `<div class="empty-state-sm">No documents indexed yet</div>`;
            return;
        }

        paperList.innerHTML = docs.map(doc => `
            <div class="paper-item">
                <span class="paper-item-name" title="${doc.source}">📄 ${doc.source}</span>
                <span class="paper-item-meta">${doc.total_chunks} chunks · ${doc.total_pages}p</span>
            </div>
        `).join("");

        // Also update chunk filter options
        const sources = ["all", ...docs.map(d => d.source)];
        chunkSourceFilter.innerHTML = sources.map(s => `
            <option value="${s}">${s === "all" ? "All Documents" : s}</option>
        `).join("");
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
    retrievalSelect.addEventListener("change", (e) => {
        state.retrievalMode = e.target.value;
        const tag = document.getElementById("pipeline-active-tag");
        if (tag) tag.textContent = `Strategy: ${e.target.options[e.target.selectedIndex].text}`;
    });

    topKSlider.addEventListener("input", (e) => {
        state.topK = parseInt(e.target.value);
        topKVal.textContent = state.topK;
    });

    topNSlider.addEventListener("input", (e) => {
        state.topN = parseInt(e.target.value);
        topNVal.textContent = state.topN;
    });

    hydeToggle.addEventListener("change", (e) => {
        state.useHyde = e.target.checked;
    });

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

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            Array.from(e.target.files).forEach(uploadFile);
        }
    });

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

    btnLoadSample.addEventListener("click", async () => {
        btnLoadSample.disabled = true;
        btnLoadSample.innerHTML = `<i data-lucide="loader" class="spin"></i> Indexing...`;
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
            btnLoadSample.innerHTML = `<i data-lucide="sparkles"></i> Load Transformer Paper`;
            lucide.createIcons();
        }
    });

    btnClearCorpus.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear all indexed documents from ChromaDB and BM25?")) return;
        try {
            await fetch("/api/clear", { method: "POST" });
            await fetchStats();
            chunksTableBody.innerHTML = `<tr><td colspan="6" class="text-center">No chunks loaded</td></tr>`;
            showToast("Knowledge base reset!");
        } catch (err) {
            alert("Error clearing corpus: " + err.message);
        }
    });

    // ==========================================
    // GROUNDED CHAT STUDIO
    // ==========================================
    async function sendChatMessage(query) {
        if (!query.trim()) return;

        // Append User Message
        appendMessage("user", query);
        chatInput.value = "";

        // Append Assistant Loading Skeleton
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
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-message ${role}-message`;
        msgDiv.innerHTML = `
            <div class="msg-avatar">
                <i data-lucide="${role === 'user' ? 'user' : 'sparkles'}"></i>
            </div>
            <div class="msg-body">
                <div class="msg-header">
                    <span class="msg-author">${role === 'user' ? 'You' : 'ScholarRAG Assistant'}</span>
                </div>
                <div class="msg-content">${marked.parse(text)}</div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        lucide.createIcons();
    }

    function appendLoadingMessage(id) {
        const msgDiv = document.createElement("div");
        msgDiv.id = id;
        msgDiv.className = "chat-message assistant-message";
        msgDiv.innerHTML = `
            <div class="msg-avatar">
                <i data-lucide="sparkles"></i>
            </div>
            <div class="msg-body">
                <div class="msg-header">
                    <span class="msg-author">ScholarRAG Assistant</span>
                    <span class="msg-tag">Retrieving & Re-Ranking...</span>
                </div>
                <div class="msg-content">
                    <p>Searching vector index, scoring with FlashRank cross-encoder, and formulating grounded citations...</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        lucide.createIcons();
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendAssistantResponse(data) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-message assistant-message";

        let citationsHtml = "";
        if (data.sources_cited && data.sources_cited.length > 0) {
            citationsHtml = `
                <div class="citations-box">
                    <span style="font-size: 0.7rem; color: var(--text-muted); display: block; width: 100%; margin-bottom: 2px;">📍 Grounded Citations (Click to view excerpt):</span>
                    ${data.sources_cited.map((s, idx) => `
                        <button class="citation-chip" data-chunk-id="${s.chunk_id || ''}" data-source="${s.source}" data-page="${s.page_number}">
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

        msgDiv.innerHTML = `
            <div class="msg-avatar">
                <i data-lucide="sparkles"></i>
            </div>
            <div class="msg-body">
                <div class="msg-header">
                    <span class="msg-author">ScholarRAG Assistant</span>
                    <span class="msg-tag">Grounded Model</span>
                </div>
                <div class="msg-content">${marked.parse(data.answer)}</div>
                ${citationsHtml}
                ${statsHtml}
            </div>
        `;

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        lucide.createIcons();

        // Attach citation click handlers
        msgDiv.querySelectorAll(".citation-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const chunkId = chip.dataset.chunkId;
                const source = chip.dataset.source;
                const page = chip.dataset.page;

                // Find corresponding chunk
                const chunk = (data.retrieved_chunks || []).find(c => c.chunk_id === chunkId) || {
                    source: source,
                    page_number: page,
                    content: "Passage context retrieved for citation."
                };
                openExcerptModal(chunk);
            });
        });
    }

    btnSendChat.addEventListener("click", () => sendChatMessage(chatInput.value));
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage(chatInput.value);
        }
    });

    promptChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const q = chip.dataset.query;
            chatInput.value = q;
            sendChatMessage(q);
        });
    });

    // ==========================================
    // MULTI-STRATEGY BENCHMARK INSPECTOR
    // ==========================================
    btnRunBenchmark.addEventListener("click", runBenchmark);
    inspectorQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") runBenchmark();
    });

    async function runBenchmark() {
        const query = inspectorQueryInput.value.trim();
        if (!query) return;

        btnRunBenchmark.disabled = true;
        btnRunBenchmark.innerHTML = `<i data-lucide="loader" class="spin"></i> Running...`;

        // Clear existing cards
        [denseCards, bm25Cards, hybridCards, rerankCards].forEach(c => {
            c.innerHTML = `<div class="empty-state">Scoring...</div>`;
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

                // 1. Dense
                denseTimer.textContent = `${(s.dense.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(denseCards, s.dense.final_chunks, "dense");

                // 2. BM25
                bm25Timer.textContent = `${(s.bm25.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(bm25Cards, s.bm25.final_chunks, "bm25");

                // 3. Hybrid RRF
                hybridTimer.textContent = `${(s.hybrid.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(hybridCards, s.hybrid.final_chunks, "hybrid");

                // 4. Re-ranker
                rerankTimer.textContent = `${(s.hybrid_rerank.retrieval_time_seconds * 1000).toFixed(1)} ms`;
                renderStrategyCards(rerankCards, s.hybrid_rerank.final_chunks, "rerank");
            }
        } catch (err) {
            alert("Benchmark failed: " + err.message);
        } finally {
            btnRunBenchmark.disabled = false;
            btnRunBenchmark.innerHTML = `<i data-lucide="play"></i> Run Comparison`;
            lucide.createIcons();
        }
    }

    function renderStrategyCards(container, chunks, type) {
        if (!chunks || chunks.length === 0) {
            container.innerHTML = `<div class="empty-state">No matching chunks found</div>`;
            return;
        }

        container.innerHTML = chunks.map((c, idx) => {
            const rank = idx + 1;
            let scoreDisplay = "";
            let badgeClass = `rank-badge rank-badge-${type}`;

            if (type === "dense") scoreDisplay = `Cosine: ${c.score.toFixed(4)}`;
            else if (type === "bm25") scoreDisplay = `BM25: ${c.score.toFixed(2)}`;
            else if (type === "hybrid") scoreDisplay = `RRF: ${c.score.toFixed(6)}`;
            else if (type === "rerank") scoreDisplay = `Re-Rank: ${c.score.toFixed(4)}`;

            return `
                <div class="retrieval-card" onclick="window.viewChunk('${encodeURIComponent(JSON.stringify(c))}')">
                    <div class="card-top-row">
                        <span class="${badgeClass}">Rank #${rank}</span>
                        <span class="score-text">${scoreDisplay}</span>
                    </div>
                    <div class="card-source-row">
                        <span>📄 ${c.source} · Page ${c.page_number}</span>
                    </div>
                    <div class="card-snippet">
                        ${c.content.substring(0, 140)}...
                    </div>
                </div>
            `;
        }).join("");
    }

    // Global helper for card clicks
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
        const query = chunkSearchInput.value.toLowerCase();
        const sourceFilter = chunkSourceFilter.value;

        const filtered = state.chunks.filter(c => {
            const matchQuery = !query || c.content.toLowerCase().includes(query) || c.chunk_id.toLowerCase().includes(query);
            const matchSource = sourceFilter === "all" || c.source === sourceFilter;
            return matchQuery && matchSource;
        });

        if (filtered.length === 0) {
            chunksTableBody.innerHTML = `<tr><td colspan="6" class="text-center">No matching chunks found</td></tr>`;
            return;
        }

        chunksTableBody.innerHTML = filtered.map(c => `
            <tr>
                <td style="font-family: var(--font-mono);">${c.chunk_index + 1}</td>
                <td style="font-family: var(--font-mono); color: var(--accent-cyan); font-size: 0.75rem;">${c.chunk_id}</td>
                <td>${c.source}</td>
                <td>Page ${c.page_number}</td>
                <td>${c.char_count}</td>
                <td style="cursor: pointer;" onclick="window.viewChunk('${encodeURIComponent(JSON.stringify(c))}')">
                    ${c.content.substring(0, 100)}...
                </td>
            </tr>
        `).join("");
    }

    chunkSearchInput.addEventListener("input", renderChunksTable);
    chunkSourceFilter.addEventListener("change", renderChunksTable);

    // ==========================================
    // MODALS
    // ==========================================
    function openSettingsModal() {
        geminiKeyInput.value = state.apiKey;
        settingsModal.classList.add("open");
    }
    function closeSettingsModal() {
        settingsModal.classList.remove("open");
    }

    btnOpenSettings.addEventListener("click", openSettingsModal);
    btnCloseSettings.addEventListener("click", closeSettingsModal);
    btnCancelSettings.addEventListener("click", closeSettingsModal);

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

    function openExcerptModal(chunk) {
        excerptModalTitle.innerHTML = `<i data-lucide="file-text"></i> ${chunk.source} (Page ${chunk.page_number})`;
        excerptModalMeta.innerHTML = `Chunk ID: <b>${chunk.chunk_id || 'N/A'}</b> | Length: <b>${chunk.content.length} chars</b> | Score: <b>${chunk.score || 'N/A'}</b>`;
        excerptModalContent.textContent = chunk.content;
        excerptModal.classList.add("open");
        lucide.createIcons();
    }
    function closeExcerptModal() {
        excerptModal.classList.remove("open");
    }
    btnCloseExcerpt.addEventListener("click", closeExcerptModal);

    // Close on backdrop click
    [settingsModal, excerptModal].forEach(modal => {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.classList.remove("open");
        });
    });

    // Toast helper
    function showToast(msg) {
        const toast = document.createElement("div");
        toast.style.position = "fixed";
        toast.style.bottom = "20px";
        toast.style.right = "20px";
        toast.style.background = "linear-gradient(135deg, var(--accent-indigo), #4F46E5)";
        toast.style.color = "#FFF";
        toast.style.padding = "10px 18px";
        toast.style.borderRadius = "8px";
        toast.style.boxShadow = "0 4px 16px rgba(0,0,0,0.5)";
        toast.style.zIndex = "9999";
        toast.style.fontSize = "0.85rem";
        toast.style.fontWeight = "600";
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3500);
    }

    // Initial Load
    fetchStats();
});

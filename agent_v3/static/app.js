// =====================================================================
//  OpenClaw Agent V3 — Frontend
//  Layout: top history drawer + search bar; main = agent answer bubble
//  + infinite-scroll product waterfall. Right side debug panel preserved.
// =====================================================================

// ---------- session / state ----------
let sessionId = localStorage.getItem("agent_v3_session_id") || "";
let userId = localStorage.getItem("agent_v3_user_id") || "";
let eventSource = null;
let eventLog = [];
let latestState = {};
let activeDebugPanel = "events";
let activeDebugView = "visual";

// History (recent searches) is derived from latestState.retrieval_history but
// also tracked locally so we get a snapshot before the first SSE final.
let historyEntries = [];
let historyOpen = false;

// Waterfall pagination state
const WATERFALL_LIMIT = 24;
let waterfallOffset = 0;
let waterfallHasMore = false;
let waterfallSeenIds = new Set();
let waterfallLoading = false;
let waterfallObserver = null;
let activeRoundId = null;

// Streaming answer
let answerStreaming = false;

// Slow-response watchdog state. `lastActivityTs` is bumped whenever any
// "real progress" SSE event arrives (token / tool / router / graph). If the
// stream goes silent for SLOW_RESPONSE_THRESHOLD_MS while we're still busy,
// a banner appears so the user knows the model is stalling and can retry.
let lastUserMessage = "";
let lastActivityTs = 0;
let slowWatchdogTimer = null;
const SLOW_RESPONSE_THRESHOLD_MS = 25000;

// Waterfall sort/filter (clientside, applied to the cached merged_stream).
// Live fetch beyond the cache uses the agent's native ranking, so when the
// user picks a non-default sort or any filter we disable infinite-scroll for
// that turn — mixing two orderings would be confusing.
let activeSort = "relevance"; // relevance | price_asc | price_desc
let activeBrands = new Set();
let priceMin = null;
let priceMax = null;
let cachedMergedStream = []; // last-known authoritative merged_stream, untouched by filters
let interestedProductIds = new Set();

// ---------- DOM refs ----------
const statusEl = document.getElementById("status");
const historyToggle = document.getElementById("historyToggle");
const historyPanel = document.getElementById("historyPanel");
const newSessionButton = document.getElementById("newSession");
const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const greetingEl = document.getElementById("greeting");
const answerBubble = document.getElementById("answerBubble");
const answerOptions = document.getElementById("answerOptions");
const waterfallToolbar = document.getElementById("waterfallToolbar");
const sortPills = document.getElementById("sortPills");
const brandPills = document.getElementById("brandPills");
const brandGroupRow = document.getElementById("brandGroupRow");
const priceMinInput = document.getElementById("priceMinInput");
const priceMaxInput = document.getElementById("priceMaxInput");
const clearFiltersBtn = document.getElementById("clearFiltersBtn");
const toolbarStatus = document.getElementById("toolbarStatus");
const waterfallEl = document.getElementById("waterfall");
const loadMoreSentinel = document.getElementById("loadMoreSentinel");
const loadMoreStatus = document.getElementById("loadMoreStatus");
const eventsPanel = document.getElementById("eventsPanel");
const eventsVisualPanel = document.getElementById("eventsVisualPanel");
const statePanel = document.getElementById("statePanel");
const stateVisualPanel = document.getElementById("stateVisualPanel");
const feedEl = document.getElementById("feed");

// ---------- helpers ----------
function setStatus(text) {
  const sessionText = sessionId ? sessionId.slice(0, 8) : "no-session";
  const userText = userId ? userId.slice(0, 12) : "no-user";
  statusEl.textContent = `${text} · ${userText} · ${sessionText}`;
}

function clearElement(element) {
  while (element && element.firstChild) element.removeChild(element.firstChild);
}

function shortJson(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function setBusy(busy) {
  input.disabled = busy;
  sendButton.disabled = busy;
  setStatus(busy ? "运行中" : "就绪");
}

// ---------- product card ----------
function productImageUrl(product) {
  const images = product?.extra?.images || [];
  const image = images.find((item) => item?.large || item?.hi_res || item?.thumb) || {};
  return image.large || image.hi_res || image.thumb || "";
}

function specItems(specs = {}) {
  const fields = [
    ["processor", "处理器"],
    ["ram", "RAM"],
    ["storage", "存储"],
    ["screen_size", "屏幕"],
    ["battery", "电池"],
    ["camera_rear", "后摄"],
    ["network", "网络"],
    ["os", "系统"],
  ];
  return fields
    .map(([key, label]) => {
      const value = specs[key];
      if (value === undefined || value === null || value === "") return null;
      return { label, value: String(value) };
    })
    .filter(Boolean)
    .slice(0, 5);
}

function compactProductForInterest(product) {
  return {
    raw_id: product.raw_id || "",
    title: product.title || "",
    price: product.price ?? null,
    rank: product.rank ?? null,
    brief_reason: product.brief_reason || "",
    score: product.score ?? null,
    match_label: product.match_label || "",
    extra: {
      brand: product.extra?.brand || null,
      specs: product.extra?.specs || {},
      images: product.extra?.images || [],
    },
  };
}

function syncInterestedFromState(state = latestState) {
  const products = Array.isArray(state?.interested_products) ? state.interested_products : [];
  interestedProductIds = new Set(products.map((item) => item?.raw_id).filter(Boolean));
  updateInterestedButtons();
}

function updateInterestedButtons() {
  // Update existing buttons' aria/title/text in place after toggle.
  document.querySelectorAll(".product-interest").forEach((button) => {
    const rawId = button.dataset.rawId || "";
    const active = !!rawId && interestedProductIds.has(rawId);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.setAttribute("aria-label", active ? "取消感兴趣" : "加入感兴趣");
    button.title = active ? "已加入感兴趣（点击取消）" : "加入感兴趣";
    button.textContent = active ? "♥" : "♡";
  });
}

async function toggleInterestedProduct(product, button = null) {
  const rawId = product?.raw_id || "";
  if (!sessionId || !rawId) return;
  const wasActive = interestedProductIds.has(rawId);
  if (button) {
    button.disabled = true;
    button.classList.add("pending");
  }
  try {
    let res;
    if (wasActive) {
      // Remove from favorites.
      const params = new URLSearchParams({ session_id: sessionId, raw_id: rawId });
      res = await fetch(`/api/products/interest?${params.toString()}`, { method: "DELETE" });
    } else {
      res = await fetch("/api/products/interest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          raw_id: rawId,
          reason: "heart_button",
          product: compactProductForInterest(product),
        }),
      });
    }
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    latestState = {
      ...latestState,
      interested_products: data.interested_products || [],
    };
    syncInterestedFromState(latestState);
    statePanel.textContent = JSON.stringify(latestState || {}, null, 2);
    renderStateVisual(latestState);
    appendEvent("product_interest", {
      action: wasActive ? "remove" : "add",
      raw_id: rawId,
      title: product.title || "",
      queue_size: (data.interested_products || []).length,
    });
  } catch (error) {
    const verb = wasActive ? "取消感兴趣" : "加入感兴趣";
    appendEvent("error", { message: `${verb}失败：${error.message || error}` });
  } finally {
    if (button) {
      button.disabled = false;
      button.classList.remove("pending");
    }
  }
}

function buildProductCard(product) {
  const card = document.createElement("article");
  card.className = "product-card";
  card.dataset.rawId = product.raw_id || "";

  const imageWrap = document.createElement("div");
  imageWrap.className = "product-image";
  const imageUrl = productImageUrl(product);
  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    img.alt = product.title || product.raw_id || "product image";
    img.loading = "lazy";
    imageWrap.appendChild(img);
  } else {
    imageWrap.textContent = "No image";
  }

  const body = document.createElement("div");
  body.className = "product-body";

  const meta = document.createElement("div");
  meta.className = "product-meta";
  const rank = document.createElement("span");
  rank.className = "product-rank";
  rank.textContent = `#${product.rank ?? "?"}`;
  meta.appendChild(rank);
  if (product.price !== null && product.price !== undefined) {
    const price = document.createElement("span");
    price.className = "product-price";
    price.textContent = `$${product.price}`;
    meta.appendChild(price);
  }
  if (product.extra?.brand) {
    const brand = document.createElement("span");
    brand.className = "product-brand";
    brand.textContent = product.extra.brand;
    meta.appendChild(brand);
  }
  if (product.match_label) {
    const match = document.createElement("span");
    match.className = "product-match";
    match.textContent = product.match_label;
    meta.appendChild(match);
  }
  if (typeof product.score === "number" && product.score > 0) {
    const score = document.createElement("span");
    score.className = "product-score";
    score.title = "rerank score";
    score.textContent = product.score.toFixed(2);
    meta.appendChild(score);
  }

  const title = document.createElement("div");
  title.className = "product-title";
  title.textContent = product.title || product.raw_id || "Untitled product";

  const specs = document.createElement("div");
  specs.className = "product-specs";
  specItems(product.extra?.specs || {}).forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "spec-chip";
    chip.textContent = `${item.label}: ${item.value}`;
    specs.appendChild(chip);
  });

  const reason = document.createElement("div");
  reason.className = "product-reason";
  reason.textContent = product.brief_reason || product.raw_id || "";

  body.appendChild(meta);
  body.appendChild(title);
  if (specs.childElementCount) body.appendChild(specs);
  if (reason.textContent) body.appendChild(reason);

  const interest = document.createElement("button");
  interest.type = "button";
  interest.className = "product-interest";
  interest.dataset.rawId = product.raw_id || "";
  const initActive = interestedProductIds.has(product.raw_id);
  if (initActive) interest.classList.add("active");
  interest.setAttribute("aria-label", initActive ? "取消感兴趣" : "加入感兴趣");
  interest.setAttribute("aria-pressed", initActive ? "true" : "false");
  interest.title = initActive ? "已加入感兴趣（点击取消）" : "加入感兴趣";
  interest.textContent = initActive ? "♥" : "♡";
  interest.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    toggleInterestedProduct(product, interest);
  });

  card.appendChild(imageWrap);
  card.appendChild(body);
  card.appendChild(interest);
  return card;
}

function appendProductsToWaterfall(products) {
  let appended = 0;
  products.forEach((product) => {
    const rawId = product.raw_id || "";
    if (!rawId || waterfallSeenIds.has(rawId)) return;
    waterfallSeenIds.add(rawId);
    waterfallEl.appendChild(buildProductCard(product));
    appended += 1;
  });
  updateInterestedButtons();
  return appended;
}

function resetWaterfall() {
  clearElement(waterfallEl);
  waterfallOffset = 0;
  waterfallHasMore = false;
  waterfallSeenIds = new Set();
  loadMoreStatus.classList.add("hidden");
  loadMoreStatus.textContent = "加载中…";
}

// ---------- waterfall sort/filter ----------
function isFiltersActive() {
  return activeSort !== "relevance" || activeBrands.size > 0 || priceMin !== null || priceMax !== null;
}

function passesFilters(product) {
  const price = typeof product.price === "number" ? product.price : null;
  if (priceMin !== null && (price === null || price < priceMin)) return false;
  if (priceMax !== null && (price === null || price > priceMax)) return false;
  if (activeBrands.size) {
    const brand = (product.extra && product.extra.brand) ? String(product.extra.brand) : "";
    if (!activeBrands.has(brand)) return false;
  }
  return true;
}

function compareForSort(a, b) {
  if (activeSort === "price_asc" || activeSort === "price_desc") {
    const pa = typeof a.price === "number" ? a.price : Number.POSITIVE_INFINITY;
    const pb = typeof b.price === "number" ? b.price : Number.POSITIVE_INFINITY;
    if (pa !== pb) return activeSort === "price_asc" ? pa - pb : pb - pa;
  }
  // Stable fallback: original rank order (which already encodes relevance score).
  return (a.rank || 0) - (b.rank || 0);
}

function applyFiltersAndSort(stream) {
  const filtered = stream.filter(passesFilters);
  if (activeSort !== "relevance") {
    filtered.sort(compareForSort);
  }
  return filtered;
}

function brandsFromStream(stream) {
  const counts = new Map();
  stream.forEach((p) => {
    const brand = p.extra && p.extra.brand;
    if (!brand) return;
    counts.set(String(brand), (counts.get(String(brand)) || 0) + 1);
  });
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function renderBrandPills() {
  if (!brandPills) return;
  clearElement(brandPills);
  const brandList = brandsFromStream(cachedMergedStream);
  if (!brandList.length) {
    brandGroupRow.classList.add("hidden");
    return;
  }
  brandGroupRow.classList.remove("hidden");
  brandList.forEach(([brand, count]) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "toolbar-pill";
    if (activeBrands.has(brand)) btn.classList.add("active");
    btn.textContent = `${brand} (${count})`;
    btn.addEventListener("click", () => {
      if (activeBrands.has(brand)) activeBrands.delete(brand);
      else activeBrands.add(brand);
      onFiltersChanged();
    });
    brandPills.appendChild(btn);
  });
}

function setSortPillActive(sort) {
  if (!sortPills) return;
  sortPills.querySelectorAll(".toolbar-pill").forEach((pill) => {
    pill.classList.toggle("active", pill.dataset.sort === sort);
  });
}

function showWaterfallToolbar(visible) {
  if (!waterfallToolbar) return;
  waterfallToolbar.classList.toggle("hidden", !visible);
}

function resetFiltersState() {
  activeSort = "relevance";
  activeBrands = new Set();
  priceMin = null;
  priceMax = null;
  if (priceMinInput) priceMinInput.value = "";
  if (priceMaxInput) priceMaxInput.value = "";
  setSortPillActive("relevance");
}

function updateToolbarStatus(visibleCount, totalCount) {
  if (!toolbarStatus) return;
  if (!totalCount) {
    toolbarStatus.textContent = "";
    return;
  }
  if (isFiltersActive()) {
    toolbarStatus.textContent =
      visibleCount === totalCount
        ? `当前 ${visibleCount} 条（已应用排序）。`
        : `符合条件 ${visibleCount} / ${totalCount} 条 · 自定义排序/筛选下不再继续加载。`;
  } else {
    toolbarStatus.textContent = `共 ${totalCount} 条，可按需排序或筛选。`;
  }
}

function onFiltersChanged() {
  // Re-render the waterfall from cache with current filters/sort.
  const filtered = applyFiltersAndSort(cachedMergedStream);
  resetWaterfall();
  appendProductsToWaterfall(filtered.slice(0, WATERFALL_LIMIT));
  waterfallOffset = Math.min(filtered.length, WATERFALL_LIMIT);
  // When filters/sort active, do NOT chain into live fetch — the live API
  // returns items in the agent's native ranking and would jumble the order.
  if (isFiltersActive()) {
    waterfallHasMore = false;
    if (filtered.length > waterfallOffset) {
      loadMoreStatus.classList.remove("hidden");
      loadMoreStatus.textContent = `还有 ${filtered.length - waterfallOffset} 条匹配项，正在显示前 ${WATERFALL_LIMIT} 条。`;
    } else {
      loadMoreStatus.classList.remove("hidden");
      loadMoreStatus.textContent = filtered.length === 0 ? "没有符合条件的商品。" : "— 没有更多了 —";
    }
  } else {
    waterfallHasMore =
      filtered.length > waterfallOffset || !!(latestState?.current_retrieval?.pagination?.has_more);
    if (!waterfallHasMore) {
      loadMoreStatus.classList.remove("hidden");
      loadMoreStatus.textContent = "— 没有更多了 —";
    } else {
      loadMoreStatus.classList.add("hidden");
    }
  }
  // Refresh brand chip active state without re-rendering counts.
  brandPills?.querySelectorAll(".toolbar-pill").forEach((pill) => {
    const label = pill.textContent || "";
    const brand = label.replace(/\s*\(\d+\)\s*$/, "");
    pill.classList.toggle("active", activeBrands.has(brand));
  });
  updateToolbarStatus(filtered.length, cachedMergedStream.length);
}

function flashProductCard(rawId) {
  if (!rawId) return;
  const card = waterfallEl.querySelector(`.product-card[data-raw-id="${CSS.escape(rawId)}"]`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.remove("flash");
  // Re-add the class on next frame so the animation re-triggers reliably.
  requestAnimationFrame(() => card.classList.add("flash"));
}

// ---------- waterfall pagination ----------
async function loadMoreProducts() {
  if (!sessionId || waterfallLoading || !waterfallHasMore) return;
  waterfallLoading = true;
  loadMoreStatus.classList.remove("hidden");
  loadMoreStatus.textContent = "加载中…";
  try {
    const params = new URLSearchParams({
      session_id: sessionId,
      offset: String(waterfallOffset),
      limit: String(WATERFALL_LIMIT),
    });
    const res = await fetch(`/api/products/stream?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const items = data.items || [];
    appendProductsToWaterfall(items);
    waterfallOffset = data.next_offset ?? waterfallOffset + items.length;
    waterfallHasMore = !!data.has_more && items.length > 0;
    if (!waterfallHasMore) {
      loadMoreStatus.textContent = items.length || waterfallSeenIds.size ? "— 没有更多了 —" : "暂无商品。";
    } else {
      loadMoreStatus.classList.add("hidden");
    }
  } catch (error) {
    loadMoreStatus.textContent = `加载失败：${error.message || error}`;
  } finally {
    waterfallLoading = false;
  }
}

function ensureWaterfallObserver() {
  if (waterfallObserver) return;
  waterfallObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          loadMoreProducts();
        }
      });
    },
    { root: feedEl, rootMargin: "200px 0px" }
  );
  waterfallObserver.observe(loadMoreSentinel);
}

// ---------- agent answer bubble (with raw_id refs) ----------
const PRODUCT_REF_RE = /\[#([A-Za-z0-9_-]+)(?:\|([^\]]+))?\]/g;

function renderAnswerInto(container, text) {
  // Replace [#raw_id|name] with clickable chips. Outside refs preserved as
  // text nodes so the streaming cursor (::after) shows up correctly.
  container.innerHTML = "";
  let lastIndex = 0;
  const source = text || "";
  let match;
  while ((match = PRODUCT_REF_RE.exec(source)) !== null) {
    const before = source.slice(lastIndex, match.index);
    if (before) container.appendChild(document.createTextNode(before));
    const rawId = match[1];
    const label = match[2] || rawId;
    const ref = document.createElement("a");
    ref.className = "product-ref";
    ref.href = "#";
    ref.dataset.rawId = rawId;
    ref.textContent = label;
    ref.title = `跳转到商品 ${rawId}`;
    ref.addEventListener("click", (event) => {
      event.preventDefault();
      flashProductCard(rawId);
    });
    container.appendChild(ref);
    lastIndex = match.index + match[0].length;
  }
  const tail = source.slice(lastIndex);
  if (tail) container.appendChild(document.createTextNode(tail));
}

function clearClarifyOptions() {
  if (!answerOptions) return;
  clearElement(answerOptions);
  answerOptions.classList.add("hidden");
  // Track which option ids are already on screen so the streaming `router_option`
  // events and the final `final` event don't render duplicates.
  answerOptions.dataset.streamedIds = "";
}

// ---------- tool trail (knowledge-lookup status above the answer) ----------
const TOOL_TRAIL_LABELS = {
  lookup_knowledge: "查询商品/品牌知识",
  retrieve_knowledge_support: "检索品类/品牌依据",
  web_lookup: "检索网络资料",
};

function ensureToolTrail() {
  let trail = document.getElementById("toolTrail");
  if (trail) return trail;
  trail = document.createElement("div");
  trail.id = "toolTrail";
  trail.className = "tool-trail hidden";
  // Insert just above the answer bubble so the "正在查询" trail
  // sits visually adjacent to the streaming reply.
  answerBubble.parentNode.insertBefore(trail, answerBubble);
  return trail;
}

function clearToolTrail() {
  closeSourcePopover();
  const trail = document.getElementById("toolTrail");
  if (!trail) return;
  clearElement(trail);
  trail.classList.add("hidden");
}

function addToolTrail(name, metadata) {
  const trail = ensureToolTrail();
  const label = TOOL_TRAIL_LABELS[name] || name;
  const query = String(metadata?.query || "").trim();
  const row = document.createElement("div");
  row.className = "tool-trail-row pending";
  row.dataset.tool = name;
  // Match by query text so multiple lookups per turn each get their own row.
  row.dataset.query = query;

  const spinner = document.createElement("span");
  spinner.className = "tool-trail-spinner";
  spinner.textContent = "🔍";
  row.appendChild(spinner);

  const text = document.createElement("span");
  text.className = "tool-trail-text";
  text.textContent = query ? `${label}：${query}` : `${label}…`;
  row.appendChild(text);

  trail.appendChild(row);
  trail.classList.remove("hidden");
  return row;
}

// ---------- slow-response banner ----------
function ensureSlowBanner() {
  let banner = document.getElementById("slowBanner");
  if (banner) return banner;
  banner = document.createElement("div");
  banner.id = "slowBanner";
  banner.className = "slow-banner hidden";
  // Sits between the tool trail and the answer bubble.
  answerBubble.parentNode.insertBefore(banner, answerBubble);
  return banner;
}

function showSlowBanner({ kind = "warning", message, showRetry = false } = {}) {
  const banner = ensureSlowBanner();
  banner.className = `slow-banner ${kind === "error" ? "is-error" : "is-warning"}`;
  clearElement(banner);
  const text = document.createElement("span");
  text.className = "slow-banner-text";
  text.textContent = message;
  banner.appendChild(text);
  if (showRetry && lastUserMessage) {
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "slow-banner-retry";
    retry.textContent = "重试";
    retry.addEventListener("click", () => {
      hideSlowBanner();
      sendMessage(lastUserMessage);
    });
    banner.appendChild(retry);
  }
}

function hideSlowBanner() {
  const banner = document.getElementById("slowBanner");
  if (!banner) return;
  banner.className = "slow-banner hidden";
  clearElement(banner);
}

function bumpActivity() {
  lastActivityTs = Date.now();
  // Any real progress hides the "slowing down" warning, but leaves any
  // hard error banner alone (those carry showRetry and stay until clicked).
  const banner = document.getElementById("slowBanner");
  if (banner && banner.classList.contains("is-warning")) hideSlowBanner();
}

function startSlowWatchdog() {
  stopSlowWatchdog();
  lastActivityTs = Date.now();
  slowWatchdogTimer = setInterval(() => {
    if (!eventSource || eventSource.readyState === 2) {
      stopSlowWatchdog();
      return;
    }
    const idleMs = Date.now() - lastActivityTs;
    if (idleMs >= SLOW_RESPONSE_THRESHOLD_MS) {
      const seconds = Math.round(idleMs / 1000);
      showSlowBanner({
        kind: "warning",
        message: `模型已 ${seconds} 秒没有新输出，可能在排队或临时拥塞，可继续等待或点重试。`,
        showRetry: true,
      });
    }
  }, 5000);
}

function stopSlowWatchdog() {
  if (slowWatchdogTimer) {
    clearInterval(slowWatchdogTimer);
    slowWatchdogTimer = null;
  }
}

function markToolFinished(name, metadata, success, durationMs) {
  const trail = document.getElementById("toolTrail");
  if (!trail) return;
  const query = String(metadata?.query || "").trim();
  // Pick the most recent pending row matching this (name, query).
  const candidates = trail.querySelectorAll(
    `.tool-trail-row.pending[data-tool="${CSS.escape(name)}"]`,
  );
  let target = null;
  for (const row of candidates) {
    if ((row.dataset.query || "") === query) target = row;
  }
  if (!target) target = candidates[candidates.length - 1] || null;
  if (!target) return;
  target.classList.remove("pending");
  target.classList.add(success ? "done" : "failed");
  const spinner = target.querySelector(".tool-trail-spinner");
  if (spinner) spinner.textContent = success ? "✓" : "⚠";
  const found = metadata?.found;
  const matchCount = metadata?.match_count;
  const status = metadata?.status;
  const ms = typeof durationMs === "number" ? Math.round(durationMs) : null;
  const tail = document.createElement("span");
  tail.className = "tool-trail-tail";
  const parts = [];
  if (found === false) parts.push("未命中");
  else if (typeof matchCount === "number" && matchCount > 0) parts.push(`${matchCount} 条`);
  else if (status) parts.push(String(status));
  if (ms !== null) parts.push(`${ms}ms`);
  tail.textContent = parts.length ? `（${parts.join(" · ")}）` : "";
  target.appendChild(tail);

  // Make the row clickable iff we have source digests to show in a popover.
  const sources = Array.isArray(metadata?.sources) ? metadata.sources : [];
  if (sources.length > 0) {
    target.classList.add("has-sources");
    const chevron = document.createElement("span");
    chevron.className = "tool-trail-chevron";
    chevron.textContent = "›";
    chevron.setAttribute("aria-hidden", "true");
    target.appendChild(chevron);
    target.setAttribute("role", "button");
    target.setAttribute("tabindex", "0");
    target.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleSourcePopover(target, sources);
    });
    target.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleSourcePopover(target, sources);
      }
    });
  }
}

// ---------- source popover (attached below a clicked tool row) ----------
let activeSourcePopover = null;

function closeSourcePopover() {
  if (!activeSourcePopover) return;
  const { popover, anchor } = activeSourcePopover;
  popover.remove();
  if (anchor) anchor.classList.remove("is-open");
  activeSourcePopover = null;
  document.removeEventListener("click", handleOutsideSourcePopover, true);
  document.removeEventListener("keydown", handleEscapeSourcePopover, true);
}

function handleOutsideSourcePopover(event) {
  if (!activeSourcePopover) return;
  const { popover, anchor } = activeSourcePopover;
  if (popover.contains(event.target) || (anchor && anchor.contains(event.target))) return;
  closeSourcePopover();
}

function handleEscapeSourcePopover(event) {
  if (event.key === "Escape") closeSourcePopover();
}

function toggleSourcePopover(anchorRow, sources) {
  if (activeSourcePopover && activeSourcePopover.anchor === anchorRow) {
    closeSourcePopover();
    return;
  }
  closeSourcePopover();
  const popover = buildSourcePopover(sources);
  // Insert as a sibling AFTER the row, inside the trail container, so the
  // flex-column layout naturally allocates space and subsequent rows shift
  // down instead of being overlapped.
  if (anchorRow.parentNode) {
    anchorRow.parentNode.insertBefore(popover, anchorRow.nextSibling);
  }
  anchorRow.classList.add("is-open");
  activeSourcePopover = { popover, anchor: anchorRow };
  // Defer outside-click listener so the click that opened us doesn't close us.
  setTimeout(() => {
    document.addEventListener("click", handleOutsideSourcePopover, true);
    document.addEventListener("keydown", handleEscapeSourcePopover, true);
  }, 0);
}

function buildSourcePopover(sources) {
  const popover = document.createElement("div");
  popover.className = "source-popover";
  const list = document.createElement("ul");
  list.className = "source-popover-list";
  for (const src of sources) {
    list.appendChild(buildSourcePopoverItem(src));
  }
  popover.appendChild(list);
  return popover;
}

function buildSourcePopoverItem(src) {
  const li = document.createElement("li");
  li.className = "source-popover-item";

  const header = document.createElement("div");
  header.className = "source-popover-header";
  const title = document.createElement("span");
  title.className = "source-popover-title";
  title.textContent = src.title || src.id || "(无标题)";
  header.appendChild(title);
  if (src.type) {
    const tag = document.createElement("span");
    tag.className = "source-popover-tag";
    tag.textContent = src.type;
    header.appendChild(tag);
  }
  li.appendChild(header);

  if (src.snippet) {
    const snippet = document.createElement("p");
    snippet.className = "source-popover-snippet";
    snippet.textContent = src.snippet;
    li.appendChild(snippet);
  }

  const refs = Array.isArray(src.sources) ? src.sources : [];
  if (refs.length) {
    const refRow = document.createElement("div");
    refRow.className = "source-popover-refs";
    for (const ref of refs) {
      let chip;
      if (ref.url) {
        chip = document.createElement("a");
        chip.href = ref.url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
      } else {
        chip = document.createElement("span");
      }
      chip.className = "source-popover-ref";
      const label = ref.publisher || ref.title || ref.id || "来源";
      chip.textContent = label;
      if (ref.url) chip.title = ref.url;
      refRow.appendChild(chip);
    }
    li.appendChild(refRow);
  }
  return li;
}

function buildClarifyOptionButton(opt) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "answer-option-btn streaming-fade-in";
  btn.dataset.optionId = String(opt.id || "");
  // <button> cannot reliably be a flex container in webkit; wrap the
  // badge + body in an inner span and put flex on that instead, otherwise
  // the description overflows below the button into the next option.
  const inner = document.createElement("span");
  inner.className = "answer-option-inner";
  const id = document.createElement("span");
  id.className = "answer-option-id";
  id.textContent = opt.id || "";
  const body = document.createElement("span");
  body.className = "answer-option-body";
  const label = document.createElement("span");
  label.className = "answer-option-label";
  label.textContent = opt.label || "";
  body.appendChild(label);
  if (opt.description) {
    const desc = document.createElement("span");
    desc.className = "answer-option-desc";
    desc.textContent = opt.description;
    body.appendChild(desc);
  }
  inner.appendChild(id);
  inner.appendChild(body);
  btn.appendChild(inner);
  btn.addEventListener("click", () => {
    const text = opt.label || opt.id || "";
    if (!text) return;
    clearClarifyOptions();
    sendMessage(text);
  });
  return btn;
}

function appendClarifyOption(opt) {
  if (!answerOptions || !opt) return;
  const optId = String(opt.id || "");
  const seen = new Set((answerOptions.dataset.streamedIds || "").split("|").filter(Boolean));
  if (optId && seen.has(optId)) return;  // dedup against streamed events
  if (optId) {
    seen.add(optId);
    answerOptions.dataset.streamedIds = Array.from(seen).join("|");
  }
  answerOptions.appendChild(buildClarifyOptionButton(opt));
  answerOptions.classList.remove("hidden");
}

function renderClarifyOptions(options) {
  if (!answerOptions) return;
  // If the streaming `router_option` events already populated the panel,
  // reuse what's there; otherwise rebuild from scratch (legacy non-stream path).
  const streamedIds = new Set((answerOptions.dataset.streamedIds || "").split("|").filter(Boolean));
  if (streamedIds.size === 0) {
    clearElement(answerOptions);
  }
  if (!Array.isArray(options) || !options.length) {
    if (streamedIds.size === 0) answerOptions.classList.add("hidden");
    return;
  }
  options.forEach((opt) => {
    const optId = String(opt.id || "");
    if (optId && streamedIds.has(optId)) return;
    if (optId) {
      streamedIds.add(optId);
      answerOptions.dataset.streamedIds = Array.from(streamedIds).join("|");
    }
    answerOptions.appendChild(buildClarifyOptionButton(opt));
  });
  answerOptions.classList.remove("hidden");
}

function startAnswerStream() {
  answerStreaming = true;
  answerBubble.dataset.text = "";
  answerBubble.classList.remove("hidden");
  answerBubble.classList.add("streaming");
  answerBubble.textContent = "";
  clearClarifyOptions();
}

function appendAnswerToken(delta) {
  if (!answerStreaming) startAnswerStream();
  const next = (answerBubble.dataset.text || "") + delta;
  answerBubble.dataset.text = next;
  renderAnswerInto(answerBubble, next);
}

function finalizeAnswer(text) {
  answerStreaming = false;
  answerBubble.classList.remove("streaming");
  if (text && text.length) {
    answerBubble.dataset.text = text;
    renderAnswerInto(answerBubble, text);
    answerBubble.classList.remove("hidden");
  } else if ((answerBubble.dataset.text || "").length === 0) {
    answerBubble.classList.add("hidden");
  }
}

// ---------- history drawer ----------
function renderHistory() {
  clearElement(historyPanel);
  if (!historyEntries.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "没有历史搜索";
    historyPanel.appendChild(empty);
    return;
  }
  historyEntries
    .slice()
    .reverse()
    .slice(0, 20)
    .forEach((entry) => {
      const item = document.createElement("div");
      item.className = "history-item";
      const text = document.createElement("span");
      text.textContent = entry.user_input || "(空)";
      const time = document.createElement("span");
      time.className = "history-item-time";
      time.textContent = entry.created_at ? entry.created_at.slice(11, 19) : "";
      item.appendChild(text);
      item.appendChild(time);
      item.addEventListener("click", () => {
        input.value = entry.user_input || "";
        historyOpen = false;
        applyHistoryOpen();
        input.focus();
      });
      historyPanel.appendChild(item);
    });
}

function applyHistoryOpen() {
  historyPanel.classList.toggle("hidden", !historyOpen);
  historyPanel.setAttribute("aria-hidden", historyOpen ? "false" : "true");
  historyToggle.setAttribute("aria-expanded", historyOpen ? "true" : "false");
  historyToggle.textContent = historyOpen ? "历史 ▴" : "历史 ▾";
}

function syncHistoryFromState(state) {
  // Build history from every human turn in state.messages — clarify rounds
  // (which produce no retrieval_history record) would otherwise vanish from
  // the sidebar. Stamp each entry with the most recent retrieval timestamp
  // available; fall back to "" when this turn was a clarify-only round.
  const messages = Array.isArray(state?.messages) ? state.messages : [];
  const retrievalRecords = Array.isArray(state?.retrieval_history) ? state.retrieval_history : [];
  const lastCreatedAt = retrievalRecords.length
    ? retrievalRecords[retrievalRecords.length - 1]?.created_at || ""
    : "";
  let humanTurnIndex = 0;
  historyEntries = messages
    .filter((message) => (message?.type || message?.role) === "human")
    .map((message) => {
      humanTurnIndex += 1;
      const matched = retrievalRecords.find(
        (record) => (record?.user_input || "") === (message?.content || "")
      );
      return {
        user_input: String(message?.content || ""),
        round_id: matched?.round_id || `turn_${humanTurnIndex}`,
        created_at: matched?.created_at || lastCreatedAt,
      };
    });
  if (historyOpen) renderHistory();
}

// ---------- debug panel helpers (kept from previous version) ----------
function textBlock(text, className = "debug-text") {
  const item = document.createElement("div");
  item.className = className;
  item.textContent = text || "";
  return item;
}

function pill(text, className = "") {
  const item = document.createElement("span");
  item.className = `debug-pill ${className}`.trim();
  item.textContent = text;
  return item;
}

function kv(label, value) {
  const row = document.createElement("div");
  row.className = "debug-kv";
  const key = document.createElement("span");
  key.className = "debug-k";
  key.textContent = label;
  const val = document.createElement("span");
  val.className = "debug-v";
  val.textContent = value === undefined || value === null || value === "" ? "—" : String(value);
  row.appendChild(key);
  row.appendChild(val);
  return row;
}

function card(title, subtitle = "") {
  const item = document.createElement("section");
  item.className = "debug-card";
  const header = document.createElement("div");
  header.className = "debug-card-header";
  const titleEl = document.createElement("div");
  titleEl.className = "debug-card-title";
  titleEl.textContent = title;
  header.appendChild(titleEl);
  if (subtitle) {
    const sub = document.createElement("div");
    sub.className = "debug-card-subtitle";
    sub.textContent = subtitle;
    header.appendChild(sub);
  }
  item.appendChild(header);
  return item;
}

function queryPlansFromState(state) {
  const queries = state?.current_retrieval?.queries;
  return Array.isArray(queries) ? queries : [];
}

function renderQueryPlanCompact(query, index) {
  const box = document.createElement("div");
  box.className = "debug-subbox";
  const planId = query.plan_id || `p${index + 1}`;
  box.appendChild(textBlock(`${planId} · ${query.query_type || "query"} · cat=${query.category || "?"}`, "debug-subtitle"));
  if (query.query) box.appendChild(kv("query", query.query));
  if (query.reason) box.appendChild(kv("reason", query.reason));
  const lexical = query.lexical_query || {};
  // New all-must design: show must / must_not. (should bucket retired, but
  // legacy plans may still have it — render if present.)
  ["must", "must_not", "should"].forEach((bucket) => {
    const values = lexical[bucket] || {};
    Object.entries(values).forEach(([field, fieldValues]) => {
      const row = document.createElement("div");
      row.className = "debug-field-row";
      row.appendChild(pill(`${bucket}.${field}`, bucket === "must" ? "field" : (bucket === "must_not" ? "negative" : "")));
      const pillsBox = document.createElement("div");
      pillsBox.className = "debug-pills";
      // Range form {min, max} — show as compact text pill.
      if (fieldValues && typeof fieldValues === "object" && !Array.isArray(fieldValues) && ("min" in fieldValues || "max" in fieldValues)) {
        const rangeText = `${fieldValues.min ?? "*"}–${fieldValues.max ?? "*"}`;
        pillsBox.appendChild(pill(rangeText, "range"));
      } else {
        const list = Array.isArray(fieldValues) ? fieldValues : [fieldValues];
        list.forEach((value) => pillsBox.appendChild(pill(String(value), bucket === "must_not" ? "negative" : "")));
      }
      row.appendChild(pillsBox);
      box.appendChild(row);
    });
  });
  // soft bucket: free-text, schema-can't-express preferences.
  const soft = query.soft || {};
  if (Object.keys(soft).length) {
    const row = document.createElement("div");
    row.className = "debug-field-row";
    row.appendChild(pill("soft", "field"));
    const pillsBox = document.createElement("div");
    pillsBox.className = "debug-pills";
    Object.entries(soft).forEach(([k, v]) => {
      if (v) pillsBox.appendChild(pill(`${k}: ${String(v).slice(0, 40)}`, "soft"));
    });
    row.appendChild(pillsBox);
    box.appendChild(row);
  }
  return box;
}

function renderRetrievalSummary(state) {
  const retrieval = state?.current_retrieval || {};
  const merged = retrieval.merged_stream || [];
  const item = card("Retrieval", retrieval.round_id || "");
  item.appendChild(kv("user_input", retrieval.user_input));
  item.appendChild(kv("query_plans", queryPlansFromState(state).length));
  item.appendChild(kv("merged_stream", merged.length));
  const pagination = retrieval.pagination || {};
  if (pagination.total_cached !== undefined) {
    item.appendChild(kv("cached", `${pagination.total_cached} (page ${pagination.page_size}, has_more=${pagination.has_more})`));
  }
  merged.slice(0, 8).forEach((product) => {
    const score = typeof product.score === "number" ? ` [${product.score.toFixed(2)}]` : "";
    item.appendChild(textBlock(`#${product.rank}${score} ${product.match_label || ""} ${product.title || product.raw_id || ""}`, "debug-note"));
  });
  return item;
}

function renderPlanSummary(state) {
  const summary = state?.current_retrieval?.plan_summary;
  if (!Array.isArray(summary) || !summary.length) return null;
  const item = card("Plan Reasons", `${summary.length} plans`);
  summary.forEach((entry) => {
    const box = document.createElement("div");
    box.className = "debug-subbox";
    const label = entry.target_label
      ? `${entry.match_label || entry.query_type || "plan"} · ${entry.target_label}`
      : (entry.match_label || entry.query_type || "plan");
    box.appendChild(textBlock(`${label} · ${entry.result_count ?? 0} 条`, "debug-subtitle"));
    if (entry.query) box.appendChild(kv("query", entry.query));
    if (entry.reason) box.appendChild(kv("reason", entry.reason));
    const support = entry.supporting_knowledge || [];
    if (support.length) {
      support.forEach((knol) => {
        const line = `${knol.source ? `[${knol.source}] ` : ""}${knol.query ? knol.query + " — " : ""}${knol.summary || ""}`;
        if (line.trim()) box.appendChild(textBlock(line, "debug-note"));
      });
    }
    item.appendChild(box);
  });
  return item;
}

function appendIntentBucket(box, label, bucket, negative = false) {
  if (!bucket || typeof bucket !== "object" || !Object.keys(bucket).length) return;
  const row = document.createElement("div");
  row.className = "debug-field-row";
  row.appendChild(pill(label, negative ? "negative" : "field"));
  const pills = document.createElement("div");
  pills.className = "debug-pills";
  Object.entries(bucket).forEach(([field, value]) => {
    pills.appendChild(pill(`${field}: ${shortJson(value)}`, negative ? "negative" : ""));
  });
  row.appendChild(pills);
  box.appendChild(row);
}

function renderCategoriesAndAction(state) {
  const item = card("Router · Categories", `type=${state?.next_action?.type || "?"}`);
  const cats = Array.isArray(state?.categories) ? state.categories : [];
  const pills = document.createElement("div");
  pills.className = "debug-pills";
  if (cats.length) {
    cats.forEach((c) => pills.appendChild(pill(c, "field")));
  } else {
    pills.appendChild(pill("(empty)", "negative"));
  }
  item.appendChild(pills);
  if (state?.next_action?.reason) item.appendChild(kv("reason", state.next_action.reason));
  if (state?.next_action?.question) item.appendChild(kv("question", state.next_action.question));
  return item;
}

function renderTraceSummary(state) {
  const trace = state?.agent_context?.trace_summary;
  if (!trace || !Object.keys(trace).length) return null;
  const item = card("Trace Summary", `${trace.event_count || 0} events · ${trace.llm_call_count || 0} LLM · ${trace.tool_call_count || 0} tool · ${trace.total_tokens || 0} tok`);
  item.appendChild(kv("turn_id", trace.turn_id));
  item.appendChild(kv("total_node_ms", formatMs(trace.total_node_ms)));
  if (trace.log_path) item.appendChild(kv("log_path", String(trace.log_path).split("/").slice(-2).join("/")));
  const byNode = trace.by_node_ms || {};
  const byTok = trace.by_node_tokens || {};
  Object.keys(byNode).forEach((node) => {
    const ms = byNode[node];
    const tok = byTok[node];
    const row = document.createElement("div");
    row.className = "debug-field-row";
    row.appendChild(pill(node, "field"));
    const pillsBox = document.createElement("div");
    pillsBox.className = "debug-pills";
    pillsBox.appendChild(pill(formatMs(ms), ms > 5000 ? "negative" : ""));
    if (tok) pillsBox.appendChild(pill(`${tok} tok`, "soft"));
    row.appendChild(pillsBox);
    item.appendChild(row);
  });
  return item;
}

function renderTopResultsByPlan(state) {
  const retrieval = state?.current_retrieval || {};
  const top = retrieval.top_results || {};
  const planKeys = Object.keys(top);
  if (!planKeys.length) return null;
  const total = planKeys.reduce((acc, k) => acc + (top[k] || []).length, 0);
  const item = card("Top Results / Plan", `${planKeys.length} plans · ${total} hits`);
  planKeys.forEach((key) => {
    const results = top[key] || [];
    const box = document.createElement("div");
    box.className = "debug-subbox";
    box.appendChild(textBlock(`${key} · ${results.length} hits`, "debug-subtitle"));
    if (!results.length) {
      box.appendChild(textBlock("(无召回)", "debug-empty"));
    } else {
      results.slice(0, 5).forEach((r) => {
        const extra = r.extra || {};
        const specs = extra.specs || {};
        const score = typeof r.score === "number" ? `[${r.score.toFixed(1)}] ` : "";
        const proc = specs.processor ? ` · ${String(specs.processor).slice(0, 24)}` : "";
        const ram = specs.ram ? ` · ${specs.ram}` : "";
        const price = r.price ? ` · $${r.price}` : "";
        box.appendChild(textBlock(`${score}${extra.brand || "?"}${proc}${ram}${price}`, "debug-note"));
        box.appendChild(textBlock(`  ${(r.title || r.raw_id || "").slice(0, 70)}`, "debug-note"));
      });
    }
    item.appendChild(box);
  });
  return item;
}

function renderKnowledgeMemory(state) {
  const km = state?.knowledge_memory || {};
  const entries = Object.entries(km);
  if (!entries.length) return null;
  const counts = { term: 0, query_support: 0, other: 0 };
  entries.forEach(([_, v]) => {
    const t = v?.memory_type;
    if (t === "term") counts.term += 1;
    else if (t === "query_support") counts.query_support += 1;
    else counts.other += 1;
  });
  const item = card("Knowledge Memory", `${entries.length} items · term=${counts.term}, query_support=${counts.query_support}`);
  // show last 3 query_support summaries
  const recent = entries
    .map(([_, v]) => v)
    .filter((v) => v?.memory_type === "query_support")
    .slice(-3);
  recent.forEach((v) => {
    const box = document.createElement("div");
    box.className = "debug-subbox";
    box.appendChild(textBlock(`${v.text || ""} · ${v.source || ""}`, "debug-subtitle"));
    if (v.summary) box.appendChild(textBlock(String(v.summary).slice(0, 200), "debug-note"));
    item.appendChild(box);
  });
  return item;
}

function renderRecallReview(state) {
  const review = state?.agent_context?.recall_review;
  if (!review || !Object.keys(review).length) return null;
  const item = card("Recall Review", `${review.status || "unknown"} · attempt ${review.attempt ?? "?"}`);
  item.appendChild(kv("reason", review.reason));
  if (review.rewrite_feedback) item.appendChild(kv("rewrite_feedback", review.rewrite_feedback));
  const violations = Array.isArray(review.violated_constraints) ? review.violated_constraints : [];
  violations.forEach((violation) => {
    const box = document.createElement("div");
    box.className = "debug-subbox";
    box.appendChild(textBlock(`${violation.target_id || "target"} · ${violation.constraint || "constraint"}`, "debug-subtitle"));
    if (violation.evidence) box.appendChild(textBlock(violation.evidence, "debug-note"));
    item.appendChild(box);
  });
  return item;
}

function renderInterestedProducts(state) {
  const products = Array.isArray(state?.interested_products) ? state.interested_products : [];
  const item = card("Interested Products", `${products.length} items`);
  if (!products.length) {
    item.appendChild(textBlock("还没有手动加入感兴趣的商品。", "debug-empty"));
    return item;
  }
  products.slice(0, 8).forEach((product) => {
    const title = product.title || product.raw_id || "";
    const rank = product.display_rank ? `#${product.display_rank} ` : "";
    const brand = product.brand ? ` · ${product.brand}` : "";
    item.appendChild(textBlock(`${rank}${title}${brand}`, "debug-note"));
  });
  return item;
}

function formatMs(value) {
  if (value === undefined || value === null || value === "") return "—";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return `${number.toFixed(number >= 100 ? 0 : 1)} ms`;
}

function formatTokens(usage = {}) {
  const total = usage.total_tokens;
  const prompt = usage.prompt_tokens;
  const completion = usage.completion_tokens;
  if (total === undefined && prompt === undefined && completion === undefined) return "—";
  return `total ${total ?? "?"} · in ${prompt ?? "?"} · out ${completion ?? "?"}`;
}

function renderRuntimeMetrics(metrics = {}) {
  const nodes = Array.isArray(metrics.nodes) ? metrics.nodes : [];
  const item = card("Runtime Metrics", `${nodes.length} nodes`);
  item.appendChild(kv("total_duration", formatMs(metrics.total_duration_ms)));
  item.appendChild(kv("total_tokens", formatTokens(metrics.token_usage || {})));
  if (!nodes.length) {
    item.appendChild(textBlock("暂无节点耗时/token 数据。", "debug-empty"));
    return item;
  }
  nodes.forEach((metric) => {
    const box = document.createElement("div");
    box.className = "debug-subbox";
    box.appendChild(textBlock(metric.node || (metric.nodes || []).join(", ") || "node", "debug-subtitle"));
    box.appendChild(kv("duration", formatMs(metric.duration_ms)));
    box.appendChild(kv("elapsed", formatMs(metric.elapsed_ms)));
    box.appendChild(kv("tokens", formatTokens(metric.token_usage || {})));
    item.appendChild(box);
  });
  return item;
}

function renderStateVisual(state) {
  clearElement(stateVisualPanel);
  if (!state || !Object.keys(state).length) {
    stateVisualPanel.appendChild(textBlock("暂无 state。", "debug-empty"));
    return;
  }
  // Order: most-debuggable first (latency, plan, recall) — operational state
  // (retrieval / interested / memory) further down.
  stateVisualPanel.appendChild(renderTraceSummary(state) || renderRuntimeMetrics(state.agent_context?.runtime_metrics || {}));
  stateVisualPanel.appendChild(renderCategoriesAndAction(state));

  const queries = queryPlansFromState(state);
  const queriesCard = card("Query Plans", `${queries.length} plans`);
  if (queries.length) {
    queries.forEach((query, index) => queriesCard.appendChild(renderQueryPlanCompact(query, index)));
  } else {
    queriesCard.appendChild(textBlock("当前 state 没有 current_retrieval.queries。", "debug-empty"));
  }
  stateVisualPanel.appendChild(queriesCard);

  const topByPlanCard = renderTopResultsByPlan(state);
  if (topByPlanCard) stateVisualPanel.appendChild(topByPlanCard);

  const reviewCard = renderRecallReview(state);
  if (reviewCard) stateVisualPanel.appendChild(reviewCard);

  if (state.current_retrieval) {
    stateVisualPanel.appendChild(renderRetrievalSummary(state));
    const planSummaryCard = renderPlanSummary(state);
    if (planSummaryCard) stateVisualPanel.appendChild(planSummaryCard);
  }

  const knowledgeCard = renderKnowledgeMemory(state);
  if (knowledgeCard) stateVisualPanel.appendChild(knowledgeCard);

  stateVisualPanel.appendChild(renderInterestedProducts(state));
}

function mergeGraphPatchIntoState(data) {
  if (!data || typeof data !== "object") return;
  Object.values(data).forEach((patch) => {
    if (!patch || typeof patch !== "object") return;
    latestState = { ...latestState, ...patch };
  });
}

function mergeDebugMetricIntoState(metric) {
  if (!metric || typeof metric !== "object") return;
  const agentContext = latestState.agent_context || {};
  const runtimeMetrics = agentContext.runtime_metrics || {};
  const nodes = [...(runtimeMetrics.nodes || []), metric];
  latestState = {
    ...latestState,
    agent_context: {
      ...agentContext,
      runtime_metrics: {
        ...runtimeMetrics,
        nodes,
        total_duration_ms: metric.elapsed_ms,
      },
    },
  };
}

function eventSummary(event) {
  if (event.name !== "graph_update") return event.name;
  const node = Object.keys(event.data || {})[0] || "graph_update";
  const patch = event.data?.[node] || {};
  const nextType = patch.next_action?.type ? ` -> ${patch.next_action.type}` : "";
  return `${node}${nextType}`;
}

function renderEventVisualItem(event) {
  const item = card(eventSummary(event), event.time);
  if (event.name === "graph_update") {
    Object.entries(event.data || {}).forEach(([node, patch]) => {
      const box = document.createElement("div");
      box.className = "debug-subbox";
      box.appendChild(textBlock(node, "debug-subtitle"));
      if (patch?.next_action?.reason) box.appendChild(kv("reason", patch.next_action.reason));
      if (patch?.shopping_context?.targets) box.appendChild(kv("targets", patch.shopping_context.targets.length));
      if (patch?.current_retrieval?.queries) box.appendChild(kv("query_plans", patch.current_retrieval.queries.length));
      if (patch?.current_retrieval?.merged_stream) box.appendChild(kv("merged_stream", patch.current_retrieval.merged_stream.length));
      if (patch?.agent_context?.recall_review) {
        const review = patch.agent_context.recall_review;
        box.appendChild(kv("recall_review", `${review.status || "unknown"} · attempt ${review.attempt ?? "?"}`));
        if (review.rewrite_feedback) box.appendChild(kv("rewrite_feedback", review.rewrite_feedback));
      }
      if (patch?.final_answer) box.appendChild(kv("answer_chars", patch.final_answer.length));
      item.appendChild(box);
    });
  } else if (event.name === "answer_token") {
    const data = event.data || {};
    item.appendChild(kv("node", data.node));
    item.appendChild(kv("delta", JSON.stringify(data.delta || "")));
  } else if (event.name === "debug_metrics") {
    const metric = event.data || {};
    item.appendChild(kv("node", metric.node || (metric.nodes || []).join(", ")));
    item.appendChild(kv("duration", formatMs(metric.duration_ms)));
    item.appendChild(kv("tokens", formatTokens(metric.token_usage || {})));
  } else if (event.name === "final") {
    const data = event.data || {};
    item.appendChild(kv("stop_reason", data.stop_reason));
    item.appendChild(kv("answer_chars", (data.answer || "").length));
    if (data.answer) {
      const preview = data.answer.length > 400 ? `${data.answer.slice(0, 400)}...` : data.answer;
      item.appendChild(textBlock(preview, "debug-note"));
    }
  } else {
    item.appendChild(textBlock(shortJson(event.data), "debug-note"));
  }
  return item;
}

function renderEventsVisual() {
  clearElement(eventsVisualPanel);
  if (!eventLog.length) {
    eventsVisualPanel.appendChild(textBlock("暂无 events。", "debug-empty"));
    return;
  }
  // answer_token events are very chatty; group consecutive ones into a single line.
  const condensed = [];
  for (const event of eventLog.slice(-160)) {
    if (event.name === "answer_token" && condensed.length) {
      const last = condensed[condensed.length - 1];
      if (last.name === "answer_token_group") {
        last.data.delta += event.data?.delta || "";
        last.data.count += 1;
        continue;
      }
    }
    if (event.name === "answer_token") {
      condensed.push({
        name: "answer_token_group",
        time: event.time,
        data: { node: event.data?.node, delta: event.data?.delta || "", count: 1 },
      });
    } else {
      condensed.push(event);
    }
  }
  condensed.forEach((event) => {
    if (event.name === "answer_token_group") {
      const item = card(`answer_token × ${event.data.count}`, event.time);
      item.appendChild(kv("node", event.data.node));
      item.appendChild(kv("preview", event.data.delta.slice(0, 200)));
      eventsVisualPanel.appendChild(item);
    } else {
      eventsVisualPanel.appendChild(renderEventVisualItem(event));
    }
  });
  eventsVisualPanel.scrollTop = eventsVisualPanel.scrollHeight;
}

function refreshDebugPanels() {
  eventsPanel.classList.toggle("hidden", activeDebugPanel !== "events" || activeDebugView !== "raw");
  eventsVisualPanel.classList.toggle("hidden", activeDebugPanel !== "events" || activeDebugView !== "visual");
  statePanel.classList.toggle("hidden", activeDebugPanel !== "state" || activeDebugView !== "raw");
  stateVisualPanel.classList.toggle("hidden", activeDebugPanel !== "state" || activeDebugView !== "visual");
}

function appendEvent(eventName, data) {
  const line = `[${new Date().toLocaleTimeString()}] ${eventName}\n${JSON.stringify(data, null, 2)}\n\n`;
  eventsPanel.textContent += line;
  eventsPanel.scrollTop = eventsPanel.scrollHeight;
  eventLog.push({ name: eventName, data, time: new Date().toLocaleTimeString() });
  if (eventName === "graph_update") {
    mergeGraphPatchIntoState(data);
    statePanel.textContent = JSON.stringify(latestState || {}, null, 2);
    renderStateVisual(latestState);
  } else if (eventName === "debug_metrics") {
    mergeDebugMetricIntoState(data);
    statePanel.textContent = JSON.stringify(latestState || {}, null, 2);
    renderStateVisual(latestState);
  }
  renderEventsVisual();
  refreshDebugPanels();
}

// ---------- session lifecycle ----------
function closeStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

async function newSession() {
  closeStream();
  if (!userId) {
    userId = `anon_${crypto.randomUUID().replaceAll("-", "")}`;
    localStorage.setItem("agent_v3_user_id", userId);
  }
  const params = new URLSearchParams({ user_id: userId });
  const res = await fetch(`/api/sessions?${params.toString()}`, { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  userId = data.user_id || userId;
  localStorage.setItem("agent_v3_session_id", sessionId);
  localStorage.setItem("agent_v3_user_id", userId);
  eventsPanel.textContent = "";
  statePanel.textContent = "";
  clearElement(eventsVisualPanel);
  clearElement(stateVisualPanel);
  eventLog = [];
  latestState = {};
  interestedProductIds = new Set();
  historyEntries = [];
  cachedMergedStream = [];
  resetFiltersState();
  showWaterfallToolbar(false);
  resetWaterfall();
  loadMoreStatus.classList.add("hidden");
  loadMoreStatus.textContent = "加载中…";
  greetingEl.classList.remove("hidden");
  answerBubble.classList.add("hidden");
  answerBubble.textContent = "";
  answerBubble.dataset.text = "";
  clearClarifyOptions();
  clearToolTrail();
  hideSlowBanner();
  stopSlowWatchdog();
  renderEventsVisual();
  renderStateVisual(latestState);
  renderHistory();
  setStatus("就绪");
}

// ---------- on a new search ----------
function onNewSearchSubmitted(message) {
  greetingEl.classList.add("hidden");
  // Do NOT reset the waterfall or hide the toolbar here. The previous
  // search's result stream stays visible while the agent thinks; it gets
  // replaced atomically when graph_update brings a new round_id (see the
  // hydrateWaterfallFromState() call in the graph_update listener). This
  // matches the design rule "瀑布流不删，只在新 query 时刷新".
  clearToolTrail();
  startAnswerStream();
  // Local optimistic history entry; will be replaced when retrieval_history syncs.
  historyEntries.push({
    user_input: message,
    round_id: `pending_${Date.now()}`,
    created_at: new Date().toISOString(),
  });
  if (historyOpen) renderHistory();
}

function sendMessage(message) {
  closeStream();
  lastUserMessage = message;
  hideSlowBanner();
  onNewSearchSubmitted(message);
  setBusy(true);
  startSlowWatchdog();

  const params = new URLSearchParams({ message });
  if (sessionId) params.set("session_id", sessionId);
  if (userId) params.set("user_id", userId);
  eventSource = new EventSource(`/api/chat/stream?${params.toString()}`);

  eventSource.addEventListener("session", (event) => {
    const data = JSON.parse(event.data);
    sessionId = data.session_id;
    userId = data.user_id || userId;
    localStorage.setItem("agent_v3_session_id", sessionId);
    localStorage.setItem("agent_v3_user_id", userId);
    appendEvent("session", data);
  });

  eventSource.addEventListener("start", (event) => appendEvent("start", JSON.parse(event.data)));
  eventSource.addEventListener("debug_metrics", (event) => appendEvent("debug_metrics", JSON.parse(event.data)));
  eventSource.addEventListener("trace_summary", (event) => {
    const data = JSON.parse(event.data);
    appendEvent("trace_summary", data);
    // Persist trace_summary onto latestState's agent_context so the
    // state-side debug card can read it before the `final` event lands.
    latestState = {
      ...latestState,
      agent_context: { ...(latestState.agent_context || {}), trace_summary: data },
    };
    renderStateVisual(latestState);
  });
  eventSource.addEventListener("graph_update", (event) => {
    const data = JSON.parse(event.data);
    appendEvent("graph_update", data);
    // When execute_query finishes, latestState gains current_retrieval.merged_stream.
    const merged = latestState?.current_retrieval?.merged_stream;
    if (Array.isArray(merged) && merged.length && latestState.current_retrieval.round_id !== activeRoundId) {
      activeRoundId = latestState.current_retrieval.round_id;
      hydrateWaterfallFromState();
    }
  });

  eventSource.addEventListener("answer_token", (event) => {
    bumpActivity();
    const data = JSON.parse(event.data);
    appendEvent("answer_token", data);
    if (data?.delta) appendAnswerToken(data.delta);
  });

  eventSource.addEventListener("router_question", (event) => {
    bumpActivity();
    const data = JSON.parse(event.data);
    appendEvent("router_question", data);
    const text = String(data?.text || "").trim();
    if (text) finalizeAnswer(text);
  });

  eventSource.addEventListener("router_option", (event) => {
    bumpActivity();
    const data = JSON.parse(event.data);
    appendEvent("router_option", data);
    if (data?.option) appendClarifyOption(data.option);
  });

  eventSource.addEventListener("tool_started", (event) => {
    bumpActivity();
    const data = JSON.parse(event.data);
    appendEvent("tool_started", data);
    addToolTrail(data?.name, data?.metadata || {});
  });

  eventSource.addEventListener("tool_finished", (event) => {
    bumpActivity();
    const data = JSON.parse(event.data);
    appendEvent("tool_finished", data);
    markToolFinished(
      data?.name,
      data?.metadata || {},
      data?.success !== false,
      data?.duration_ms,
    );
  });

  eventSource.addEventListener("final", (event) => {
    const data = JSON.parse(event.data);
    appendEvent("final", data);
    latestState = data.state || {};
    syncInterestedFromState(latestState);
    statePanel.textContent = JSON.stringify(data.state || {}, null, 2);
    renderStateVisual(latestState);
    finalizeAnswer(data.answer || "");
    const nextAction = latestState?.next_action || {};
    if (data.needs_clarification && Array.isArray(nextAction.options) && nextAction.options.length) {
      renderClarifyOptions(nextAction.options);
    } else {
      clearClarifyOptions();
    }
    syncHistoryFromState(latestState);
    if (Array.isArray(latestState?.current_retrieval?.merged_stream)) {
      // Final hydrate ensures waterfall reflects the authoritative state.
      hydrateWaterfallFromState();
    }
    closeStream();
    setBusy(false);
    stopSlowWatchdog();
    hideSlowBanner();
    input.focus();
  });

  eventSource.addEventListener("error", (event) => {
    let data = { message: "stream error" };
    if (event.data) {
      try {
        data = JSON.parse(event.data);
      } catch (_) {
        data = { message: event.data };
      }
    }
    appendEvent("error", data);
    closeStream();
    setBusy(false);
    stopSlowWatchdog();
    const isTimeout = data?.kind === "timeout"
      || /timeout|timed out|readtimeout/i.test(String(data?.message || ""));
    showSlowBanner({
      kind: "error",
      message: isTimeout
        ? "模型响应超时（可能是接口拥塞），可点击重试。"
        : `请求出错：${data?.message || "未知错误"}`,
      showRetry: true,
    });
  });
}

function hydrateWaterfallFromState() {
  const retrieval = latestState?.current_retrieval || {};
  const merged = retrieval.merged_stream || [];
  syncInterestedFromState(latestState);
  if (!merged.length) {
    cachedMergedStream = [];
    showWaterfallToolbar(false);
    return;
  }
  cachedMergedStream = merged;
  // New round → drop any prior filters (brand chips from a previous turn would
  // be stale anyway) and rebuild the brand pill list.
  resetFiltersState();
  renderBrandPills();
  showWaterfallToolbar(true);
  // Reset (so re-hydrating after final doesn't double-add) and replay.
  resetWaterfall();
  appendProductsToWaterfall(merged.slice(0, WATERFALL_LIMIT));
  waterfallOffset = Math.min(merged.length, WATERFALL_LIMIT);
  waterfallHasMore = merged.length > waterfallOffset || !!retrieval.pagination?.has_more;
  if (!waterfallHasMore) {
    loadMoreStatus.classList.remove("hidden");
    loadMoreStatus.textContent = "— 没有更多了 —";
  } else {
    loadMoreStatus.classList.add("hidden");
  }
  updateToolbarStatus(merged.length, merged.length);
  ensureWaterfallObserver();
}

// ---------- DOM events ----------
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  sendMessage(message);
});

newSessionButton.addEventListener("click", () => {
  newSession().catch((error) => appendEvent("error", { message: String(error) }));
});

historyToggle.addEventListener("click", () => {
  historyOpen = !historyOpen;
  applyHistoryOpen();
  if (historyOpen) renderHistory();
});

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    activeDebugPanel = button.dataset.panel || "events";
    refreshDebugPanels();
  });
});

document.querySelectorAll(".view-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    activeDebugView = button.dataset.view || "visual";
    refreshDebugPanels();
  });
});

// Toolbar: sort pills
sortPills?.querySelectorAll(".toolbar-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    activeSort = pill.dataset.sort || "relevance";
    setSortPillActive(activeSort);
    onFiltersChanged();
  });
});

// Toolbar: price range — debounce by waiting for `change`/blur not `input` so
// half-typed numbers don't thrash the waterfall.
function syncPriceFromInputs() {
  const minRaw = priceMinInput?.value;
  const maxRaw = priceMaxInput?.value;
  priceMin = minRaw && !Number.isNaN(Number(minRaw)) ? Number(minRaw) : null;
  priceMax = maxRaw && !Number.isNaN(Number(maxRaw)) ? Number(maxRaw) : null;
  onFiltersChanged();
}
priceMinInput?.addEventListener("change", syncPriceFromInputs);
priceMaxInput?.addEventListener("change", syncPriceFromInputs);
priceMinInput?.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); syncPriceFromInputs(); } });
priceMaxInput?.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); syncPriceFromInputs(); } });

clearFiltersBtn?.addEventListener("click", () => {
  resetFiltersState();
  onFiltersChanged();
});

// ---------- bootstrap ----------
renderEventsVisual();
renderStateVisual(latestState);
refreshDebugPanels();
applyHistoryOpen();
setStatus(sessionId ? "就绪" : "未创建会话");

"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const request = (url, options = {}) => fetch(url, {signal:AbortSignal.timeout(15000), ...options});
  const number = (v) => new Intl.NumberFormat("zh-CN").format(v || 0);
  const duration = (v) => v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
  const p95 = (v) => v.p95_ms == null ? "—" : `${v.p95_overflow ? ">" : "≤"} ${duration(v.p95_ms)}`;
  const date = (v) => v ? new Intl.DateTimeFormat("zh-CN", {timeZone:"Asia/Shanghai", month:"2-digit", day:"2-digit", hour:"2-digit", minute:"2-digit", hour12:false}).format(new Date(v)) : "—";
  let csrf = null, snapshot = null, expiry = null, loading = false;
  const keysPage = location.pathname.replace(/\/$/, "") === "/admin/api-keys";
  document.querySelector(`[data-nav="${keysPage ? "keys" : "overview"}"]`).setAttribute("aria-current", "page");
  $("overview").hidden = keysPage;
  $("keys").hidden = !keysPage;

  function notice(text) { $("status").textContent = text; $("status").hidden = !text; }
  function clearPrivateView() {
    csrf = null; snapshot = null; clearTimeout(expiry);
    $("dashboard").hidden = true; $("account").hidden = true; $("login").hidden = false;
    for (const id of ["ui-count","rest-count","mcp-count","ui-detail","rest-detail","mcp-detail","ui-avg","ui-p95","window-count","window-errors","window-status","window-range","latency-note","since","updated","username"]) $(id).textContent = "—";
    for (const id of ["endpoints","latency-chart","trend-chart"]) $(id).replaceChildren();
    for (const id of ["history-state","history-insight","history-stats","history-chart","history-note","quality-stats","rest-insight","window-slow","sources","clients","pressure-stats"]) $(id).replaceChildren();
  }
  async function session() {
    const response = await request("/admin/api/session", {cache:"no-store", credentials:"same-origin"});
    const data = await response.json();
    if (!response.ok || !data.authenticated) {
      clearPrivateView();
      $("login-link").hidden = !data.configured;
      if (data.login_url) $("login-link").href = data.login_url;
      $("login-message").textContent = data.configured ? "登录将在新标签页打开。退出后台不会退出你的 Hugging Face 账号。" : "后台登录尚未配置。部署到目标 HF Space 并启用 OAuth 后即可验证，普通搜索不受影响。";
      return false;
    }
    csrf = data.csrf;
    $("username").textContent = data.username;
    $("account").hidden = false; $("login").hidden = true; $("dashboard").hidden = false;
    clearTimeout(expiry);
    expiry = setTimeout(() => {clearPrivateView(); notice("后台登录已过期，请重新登录。"); session().catch(() => {});}, data.expires_in * 1000);
    return true;
  }
  function svgNode(name, attrs = {}, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key, value));
    if (text != null) node.textContent = text;
    return node;
  }
  function chart(id, items, trend = false) {
    const root = $(id); root.replaceChildren();
    if (!items.length || !items.some((item) => item.count > 0)) {
      const p = document.createElement("p"); p.className = "chart-empty"; p.textContent = "暂无可用观测数据"; root.append(p); return;
    }
    const width = Math.max(280, root.clientWidth || 480), height = trend ? 210 : 125, left = 38, bottom = 30;
    const top = 18, plot = height - bottom - top;
    const max = Math.max(...items.map((item) => item.count || 0), 1);
    const step = (width - left - 10) / items.length;
    const chartTitle = id === "history-chart" ? "事件循环平均延迟（毫秒）" : trend ? "REST 每小时请求趋势" : "UI 搜索延迟分布";
    const svg = svgNode("svg", {viewBox:`0 0 ${width} ${height}`, role:"img", "aria-label":chartTitle});
    svg.append(svgNode("title", {}, chartTitle));
    [0, .5, 1].forEach((ratio) => {
      const y = top + plot * (1 - ratio);
      svg.append(svgNode("line", {x1:left,y1:y,x2:width-10,y2:y,class:"gridline"}));
      svg.append(svgNode("text", {x:left-8,y:y+4,"text-anchor":"end"}, number(Math.round(max * ratio))));
    });
    items.forEach((item,i) => {
      const x = left + i * step, barHeight = plot * (item.count || 0) / max;
      if (item.count != null) {
        if (trend && item.endpoints) {
          let used = 0;
          for (const endpoint of ["search","related","artists","health"]) {
            const count = item.endpoints[endpoint] || 0, h = plot * count / max;
            const rect = svgNode("rect", {x:x+1,y:top+plot-used-h,width:Math.max(.5,step-1),height:h,class:`endpoint-${endpoint}`});
            rect.append(svgNode("title", {}, `${item.label} · ${endpoint}: ${number(count)} 次`));svg.append(rect);used+=h;
          }
        } else {
          const rect = svgNode("rect", {x:x+1,y:top+plot-barHeight,width:Math.max(.5,step-2),height:barHeight,rx:1,class:trend?"trend-bar":"latency-bar"});
          rect.append(svgNode("title", {}, item.tooltip || `${item.label}: ${number(item.count)} 次`)); svg.append(rect);
        }
      } else svg.append(svgNode("line", {x1:x+step/2,y1:top+plot-3,x2:x+step/2,y2:top+plot,class:"missing"}));
      const labelCount = width < 450 ? 3 : trend ? 5 : 6;
      if (i % Math.max(1,Math.ceil(items.length / labelCount)) === 0) svg.append(svgNode("text", {x:x+2,y:height-8}, item.short || item.label));
    });
    root.append(svg);
  }
  function trend() {
    if (!snapshot) return;
    const rest = snapshot.rest_windows[$("range").value];
    const items = rest.hours.map((item) => ({count:item.count,endpoints:item.endpoints,label:item.hour.replace("T"," "),short:item.hour.slice(5,10) + " " + item.hour.slice(11,13)+"时"}));
    chart("trend-chart", items, true);
  }
  const percent = (value) => value == null ? "—" : `${value}%`;
  function signals(id, items) {
    const root = $(id);root.replaceChildren();
    items.forEach(([label,value]) => {const box=document.createElement("div"),caption=document.createElement("span"),amount=document.createElement("strong");caption.textContent=label;amount.textContent=value;box.append(caption,amount);root.append(box);});
  }
  function ranks(id, items, total) {
    const root=$(id);root.replaceChildren();
    if(!items.length){root.textContent="暂无观测记录";return;}
    items.forEach((item) => {const row=document.createElement("div"),label=document.createElement("span"),value=document.createElement("strong"),meter=document.createElement("meter");label.textContent=item.label;value.textContent=`${number(item.count)} · ${total ? (100*item.count/total).toFixed(1) : 0}%`;meter.min=0;meter.max=total || 1;meter.value=item.count;meter.setAttribute("aria-label",`${item.label} 请求占比`);row.append(label,value,meter);root.append(row);});
  }
  function renderHistory() {
    const h=snapshot.ui_performance;
    $("history-state").textContent=h.stale?"采样已过期":`${h.window_count} 个性能窗口`;
    $("history-stats").replaceChildren();
    $("history-chart").replaceChildren();
    $("history-note").textContent="仅显示当前进程最近最多 180 个性能采样窗口，重启后重新积累；不是 UI 搜索耗时，也不代表历史搜索量。无需配置或上传额外文件。";
    if(!h.latest){
      $("history-insight").textContent="当前进程尚无 UI 性能采样，产生采样后自动显示。累计统计及 REST 观测仍可使用。";
      return;
    }
    const latest=h.latest, metrics=latest.metrics || {};
    $("history-insight").textContent=`最近采样 ${date(latest.recorded_at)}。${h.stale ? "已超过 5 分钟未更新，请勿当作当前状态。" : "显示事件循环及界面响应的实际观测。"}`;
    signals("history-stats",[["事件循环平均延迟",duration(metrics.event_loop_lag?.avg_ms)],["浏览器往返平均耗时",duration(metrics.browser_roundtrip?.avg_ms)],["推荐等待平均耗时",duration(metrics.recommendation_wait?.avg_ms)],["推荐计算平均耗时",duration(metrics.recommendation_compute?.avg_ms)]]);
    chart("history-chart",h.lag_trend.map((item)=>({count:item.average_ms,label:date(item.at),short:date(item.at),tooltip:`${date(item.at)} 事件循环平均延迟 ${duration(item.average_ms)}`})),true);
  }
  function renderRest() {
    if(!snapshot)return;
    const rest=snapshot.rest_windows[$("range").value], label=$("range").selectedOptions[0].textContent;
    document.querySelectorAll(".window-label").forEach((node)=>{node.textContent=label;});
    $("window-count").textContent=number(rest.count);
    $("window-errors").textContent=`${number(rest.statuses["4xx"])} / ${number(rest.statuses["5xx"])}`;
    $("window-slow").textContent=`${number(rest.slow_count)} · ${percent(rest.slow_percent)}`;
    $("window-status").textContent=["2xx","3xx","4xx","5xx"].map((key)=>`${key} ${number(rest.statuses[key])}`).join(" / ");
    $("window-range").textContent=rest.first_hour?`记录 ${rest.first_hour.replace("T"," ")} — ${rest.last_hour.replace("T"," ")} · ${rest.observed_hours} / ${rest.requested_hours} 个小时有记录` : "当前窗口没有 REST 观测记录。";
    $("rest-insight").textContent=rest.count?`${label}：${percent(rest.slow_percent)} 的已记录请求耗时超过 5 秒；HTTP 在途峰值 ${number(rest.peak_in_flight)}。4xx ${number(rest.statuses["4xx"])} 次与 5xx ${number(rest.statuses["5xx"])} 次分别统计，不将全部错误视为服务器故障。`:"当前窗口没有记录；时间空白不等于零流量。";
    const tbody=$("endpoints");tbody.replaceChildren();
    rest.endpoints.forEach((row)=>{const tr=document.createElement("tr");[`/api/${row.endpoint}${row.endpoint==="health"?"（健康检查）":""}`,number(row.count),duration(row.average_ms),p95(row),percent(row.slow_percent),number(row.client_errors),number(row.server_errors)].forEach((value)=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});tbody.append(tr);});
    if(!rest.endpoints.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=7;td.textContent="暂无接口观测数据";tr.append(td);tbody.append(tr);}
    ranks("sources",rest.sources,rest.count);ranks("clients",rest.clients,rest.count);
    signals("pressure-stats",[["HTTP 在途峰值",rest.peak_in_flight == null?"—":number(rest.peak_in_flight)],["分组分钟请求峰值",rest.peak_per_minute == null?"—":number(rest.peak_per_minute)],["有记录的小时",`${rest.observed_hours} / ${rest.requested_hours}`]]);
    trend();
  }
  function render(data) {
    snapshot = data;
    const c = data.counters, sum = (names) => names.reduce((total,key) => total + (c[key] || 0),0);
    $("updated").textContent = `快照时间 ${date(data.generated_at)}`;
    $("since").textContent = `累计 · 自 ${data.telemetry_since ? data.telemetry_since.slice(0,10) : "开始采集"}`;
    $("ui-count").textContent = number(c.ui_search);
    $("ui-detail").textContent = `${number(c.ui_visit)} 次访问 · ${number(c.ui_zero_result)} 次零结果`;
    $("rest-count").textContent = number(sum(["rest_search","rest_related","rest_artists"]));
    $("rest-detail").textContent = `搜索 ${number(c.rest_search)} · 关联 ${number(c.rest_related)} · 画师 ${number(c.rest_artists)}`;
    $("mcp-count").textContent = number(sum(Object.keys(c).filter((key) => key.startsWith("mcp_"))));
    $("mcp-detail").textContent = `标签搜索 ${number(c.mcp_search_tags)} · 相关标签 ${number(c.mcp_get_related_tags)}`;
    $("ui-avg").textContent = duration(data.ui_latency.average_ms);
    $("ui-p95").textContent = p95(data.ui_latency);
    $("latency-note").textContent = `${number(data.ui_latency.count)} 个延迟样本。${data.ui_latency.distribution_available ? "分桶分别计数，P95 为区间估算；超过末档时显示 > 120 s。" : "当前数据未提供完整分桶，不能还原分布或估算 P95。"}`;
    chart("latency-chart", data.ui_latency.buckets);
    const q=data.quality;
    signals("quality-stats",[["发生选词的搜索会话",percent(q.selection_percent)],["复制操作 / 搜索",percent(q.copy_events_per_search)],["零结果率",percent(q.zero_percent)],["60 秒内重复搜索",percent(q.repeat_percent)],["冷启动成功 / 尝试",`${number(q.cold_successes)} / ${number(q.cold_attempts)}`],["冷启动失败",number(q.cold_failures)]]);
    renderHistory();renderRest();
  }
  async function refresh() {
    if (loading || document.hidden) return;
    loading = true; $("refresh").disabled = true;
    try {
      if (!await session()) return;
      if (keysPage) return;
      const response = await request("/admin/api/overview", {cache:"no-store",credentials:"same-origin"});
      if (response.status === 401 || response.status === 403) {clearPrivateView();notice("后台登录已失效，请重新登录。");await session();return;}
      if (!response.ok) throw new Error("metrics");
      render(await response.json()); notice("");
    } catch (_) {notice(snapshot ? "刷新失败，当前保留上一次快照。请稍后重试，并注意快照时间。" : "暂时无法读取后台状态，请稍后刷新。标签搜索可从左侧入口访问。");}
    finally {loading=false;$("refresh").disabled=false;}
  }
  $("refresh").addEventListener("click", refresh);
  $("range").addEventListener("change", renderRest);
  $("logout").addEventListener("click", async () => {
    $("logout").disabled=true;
    try {
      const response = await request("/admin/logout", {method:"POST",credentials:"same-origin",headers:{"X-CSRF-Token":csrf || ""}});
      if (!response.ok && response.status !== 401) throw new Error("logout");
      clearPrivateView(); location.replace("/admin?notice=logged_out");
    } catch (_) {notice("退出未完成，请重试。当前登录状态尚未确认失效。");}
    finally {$("logout").disabled=false;}
  });
  const keyTabs = {applications:["申请审核尚未启用","后续在这里查看账号、服务名称、用途和预计用量，并批准或拒绝申请。"],grants:["授权与 Key 管理尚未启用","后续在这里查看脱敏 Key、授权范围、用量和有效期，并暂停、恢复或吊销授权。"],audit:["操作记录尚未启用","后续在这里查看审批、授权调整与吊销记录。当前没有启用授权操作审计。"]};
  const tabs = [...document.querySelectorAll("[data-key-tab]")];
  tabs.forEach((button, i) => {
    button.addEventListener("click", () => {
      tabs.forEach((tab) => {tab.setAttribute("aria-selected", String(tab===button));tab.tabIndex=tab===button?0:-1;});
      const [title,copy]=keyTabs[button.dataset.keyTab];$("key-title").textContent=title;$("key-copy").textContent=copy;$("key-panel").setAttribute("aria-labelledby",button.id);
    });
    button.addEventListener("keydown", (event) => {if (["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) {event.preventDefault();const next=event.key==="Home"?0:event.key==="End"?tabs.length-1:(i+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length;tabs[next].click();tabs[next].focus();}});
  });
  const messages={forbidden:"HF 身份验证成功，但该账号没有后台管理员权限。请使用指定管理员账号登录。",login_failed:"登录验证未完成或已过期，请重新发起登录。",busy:"登录请求较多，请几分钟后重试。",unconfigured:"后台 OAuth 尚未配置，暂时不能登录。",logged_out:"已退出后台。"};
  const message=messages[new URLSearchParams(location.search).get("notice")];if(message) notice(message);
  // Avoid displaying previous private DOM when restoring a browser history entry.
  window.addEventListener("pagehide", clearPrivateView);
  window.addEventListener("pageshow", (event) => {if(event.persisted) refresh();});
  document.addEventListener("visibilitychange", () => {if(!document.hidden) refresh();});
  setInterval(refresh, 30000);
  let resizeTimer;
  window.addEventListener("resize", () => {clearTimeout(resizeTimer);resizeTimer=setTimeout(() => {if(snapshot && !keysPage) {chart("latency-chart",snapshot.ui_latency.buckets);renderHistory();trend();}},100);});
  refresh();
})();

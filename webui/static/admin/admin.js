"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const request = (url, options = {}) => fetch(url, {signal:AbortSignal.timeout(15000), ...options});
  const number = (v) => new Intl.NumberFormat("zh-CN").format(v || 0);
  const duration = (v) => v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
  const p95 = (v) => v.p95_ms == null ? "—" : `${v.p95_overflow ? ">" : "≤"} ${duration(v.p95_ms)}`;
  const date = (v) => v ? new Intl.DateTimeFormat("zh-CN", {timeZone:"Asia/Shanghai", month:"2-digit", day:"2-digit", hour:"2-digit", minute:"2-digit", hour12:false}).format(new Date(v)) : "—";
  let csrf = null, snapshot = null, expiry = null, loading = false, detailPage = 0, detailRows = [];
  const eventLabels = {ui_visit:"网页访问",ui_search:"网页搜索",ui_zero_result:"零结果搜索",ui_search_with_selection_session:"发生选词的搜索会话",ui_repeat_search_60s:"60 秒内重复搜索",ui_copy_selected:"复制已选标签",ui_copy_all:"复制全部标签",rest_search:"REST 搜索",rest_related:"REST 相关标签",rest_artists:"REST 画师推荐",mcp_search_tags:"MCP 搜索标签",mcp_get_related_tags:"MCP 相关标签",mcp_get_artist_recommendations:"MCP 画师推荐",mcp_get_artist_profile:"MCP 画师档案",mcp_get_anima_format:"MCP Anima 格式",mcp_get_newbie_format:"MCP NewBie 格式",mcp_get_qwen_image_2_1_format:"MCP Qwen Image 格式",feedback_search_bad_case:"搜索问题反馈",feedback_translation_error:"翻译错误反馈",engine_cold_start_attempt:"冷启动尝试",engine_cold_start_success:"冷启动成功",engine_cold_start_failure:"冷启动失败"};
  const timingLabels = {ui_search_latency:"网页搜索耗时",search_to_first_selection:"搜索至首次选词",search_to_first_copy:"搜索至首次复制",engine_cold_start:"引擎冷启动"};
  const performanceLabels = {event_loop_lag:"事件循环延迟",browser_roundtrip:"浏览器往返",recommendation_wait:"推荐等待",recommendation_compute:"推荐计算",selected_render:"已选标签渲染",history_render:"历史记录渲染",favorites_render:"收藏渲染",related_render:"相关标签渲染",artist_render:"画师渲染",group_render:"标签分组渲染",group_page_render:"分组翻页渲染"};
  const reasonLabels = {success:"成功",redirect:"重定向",anonymous_rate_limited:"匿名池 · 频率限制",anonymous_concurrency_limited:"匿名池 · 并发限制",key_rate_limited:"Key · 频率限制",key_concurrency_limited:"Key · 并发限制",rest_rate_limited:"服务整体 · 频率限制",rest_concurrency_limited:"服务整体 · 并发限制",authentication_rate_limited:"认证前置限流",anonymous_daily_quota_exhausted:"匿名池日额度不足",key_daily_quota_exhausted:"Key 日额度不足",invalid_api_key:"无效 Key",registration_mismatch:"Client / Site 登记不匹配",preview_account_required:"账号无预览权限",validation_json:"JSON 格式错误",validation_missing:"缺少必填参数",validation_literal:"参数枚举错误",validation_bounds:"参数超出范围",validation_type:"参数类型错误",validation_other:"其他参数校验错误",not_found:"接口不存在",method_not_allowed:"请求方法错误",client_error_other:"其他客户端错误",server_error:"服务端错误",legacy_unknown:"历史原因未采集"};
  const reasonName = (key) => reasonLabels[key] || key;
  const latencyBucket = (key) => {
    const limits=[100,250,500,1000,2000,5000,10000,30000,60000,120000];
    if(key==="gt_120000")return "> 120 s";
    const index=limits.findIndex(limit=>key===`le_${limit}`);
    return index<0?(key || "—"):`${index?duration(limits[index-1]):"0 ms"}–${duration(limits[index])}`;
  };
  const presentNumber = (v) => v == null ? "—" : number(v);
  const bytes = (v) => v == null ? "—" : `${(v / 1024 / 1024).toFixed(1)} MiB`;
  const sourceName = (r) => r.source_kind === "anonymous" ? `未声明 · ${r.source_name || r.client_family || "unknown"}` : r.source_kind === "declared" ? `声明 · ${r.source_name || "未命名"}${r.source_site ? " · " + r.source_site : ""}` : [r.source_kind || "无来源维度", r.source_name, r.source_site].filter(Boolean).join(" · ");
  function selectedRest() {
    const window = snapshot.rest_windows[$("range").value];
    return $("rest-scope").value === "business" ? window.business : window;
  }
  function table(id, rows, columns, empty = "当前范围没有可用记录") {
    const root=$(id);root.replaceChildren();
    rows.forEach(values => {const tr=document.createElement("tr");values.forEach(value => {const td=document.createElement("td");td.textContent=value;tr.append(td);});root.append(tr);});
    if(!rows.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=columns;td.textContent=empty;tr.append(td);root.append(tr);}
  }
  function options(id, values, label, name = (value) => value) {
    const select=$(id),previous=select.value;select.replaceChildren();
    if(label != null){const option=document.createElement("option");option.value="";option.textContent=label;select.append(option);}
    values.forEach(value=>{const option=document.createElement("option");option.value=value;option.textContent=name(value);select.append(option);});
    if([...select.options].some(option=>option.value===previous))select.value=previous;
  }
  let keysPage = location.pathname.replace(/\/$/, "") === "/admin/api-keys";
  document.querySelector(`[data-nav="${keysPage ? "keys" : "overview"}"]`).setAttribute("aria-current", "page");
  $("overview").hidden = keysPage;
  $("keys").hidden = !keysPage;

  function notice(text) { $("status").textContent = text; $("status").hidden = !text; }
  function clearPrivateView() {
    window.keyPortal?.clear();
    csrf = null; snapshot = null; clearTimeout(expiry);
    detailRows=[];detailPage=0;
    $("dashboard").hidden = true; $("account").hidden = true; $("login").hidden = false;
    for (const id of ["ui-count","rest-count","mcp-count","ui-detail","rest-detail","mcp-detail","ui-avg","ui-p95","window-count","window-errors","window-status","window-range","latency-note","since","updated","username"]) $(id).textContent = "—";
    for (const id of ["endpoints","latency-chart","trend-chart"]) $(id).replaceChildren();
    for (const id of ["history-state","history-insight","history-stats","history-chart","history-note","quality-stats","rest-insight","window-slow","sources","clients","pressure-stats"]) $(id).replaceChildren();
    for (const id of ["data-boundary","runtime-stats","performance-metrics","gc-metrics","counter-details","timing-details","timing-chart","feedback-summary","rejection-stats","status-codes","outcome-reasons","error-trend","hourly-details","diagnostic-details","detail-summary","detail-page","performance-window","timing-metric","detail-endpoint","detail-reason","detail-status","detail-client"]) $(id).replaceChildren();
    $("detail-source").value="";
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
    if (!items.length || !items.some((item) => item.count != null)) {
      const p = document.createElement("p"); p.className = "chart-empty"; p.textContent = "暂无可用观测数据"; root.append(p); return;
    }
    const width = Math.max(280, root.clientWidth || 480), height = trend ? 210 : 125, left = 38, bottom = 30;
    const top = 18, plot = height - bottom - top;
    const max = Math.max(...items.map((item) => item.count || 0), 1);
    const step = (width - left - 10) / items.length;
    const chartTitle = id === "history-chart" ? "事件循环平均延迟（毫秒）" : id === "error-trend" ? "每小时拒绝与错误：429、其他 4xx、5xx" : trend ? "REST 每小时请求趋势" : "耗时分布（各区间分别计数）";
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
        if (trend && (item.endpoints || item.stacks)) {
          let used = 0;
          const keys = item.stacks ? ["rejected","client-error","server-error"] : ["search","related","artists","health"];
          for (const endpoint of keys) {
            const count = (item.stacks || item.endpoints)[endpoint] || 0, h = plot * count / max;
            const rect = svgNode("rect", {x:x+1,y:top+plot-used-h,width:Math.max(.5,step-1),height:h,class:`endpoint-${endpoint}`});
            rect.append(svgNode("title", {}, `${item.label} · ${{rejected:"429", "client-error":"其他 4xx", "server-error":"5xx"}[endpoint] || endpoint}: ${number(count)} 次`));svg.append(rect);used+=h;
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
    const rest = selectedRest();
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
    const windows=h.windows || [];
    options("performance-window",["latest",...windows.slice().reverse().map(item=>item.recorded_at)],null,value=>value==="latest"?"最新窗口":date(value));
    renderPerformanceDetails();
    if(!h.latest){
      $("history-insight").textContent="当前进程尚无 UI 性能采样，产生采样后自动显示。累计统计及 REST 观测仍可使用。";
      return;
    }
    const latest=h.latest, metrics=latest.metrics || {};
    $("history-insight").textContent=`最近采样 ${date(latest.recorded_at)}。${h.stale ? "已超过 5 分钟未更新，请勿当作当前状态。" : "显示事件循环及界面响应的实际观测。"}`;
    signals("history-stats",[["事件循环平均延迟",duration(metrics.event_loop_lag?.avg_ms)],["浏览器往返平均耗时",duration(metrics.browser_roundtrip?.avg_ms)],["推荐等待平均耗时",duration(metrics.recommendation_wait?.avg_ms)],["推荐计算平均耗时",duration(metrics.recommendation_compute?.avg_ms)]]);
    chart("history-chart",h.lag_trend.map((item)=>({count:item.average_ms,label:date(item.at),short:date(item.at),tooltip:`${date(item.at)} 事件循环平均延迟 ${duration(item.average_ms)}`})),true);
  }
  function renderPerformanceDetails() {
    if(!snapshot)return;
    const h=snapshot.ui_performance,window=$("performance-window").value==="latest"?h.latest:(h.windows || []).find(item=>item.recorded_at===$("performance-window").value);
    const r=window?.runtime || {};
    signals("runtime-stats",[["进程 CPU（单核 = 100%）",percent(r.process_cpu_percent)],["RSS 内存",bytes(r.rss_bytes)],["逻辑核数",presentNumber(r.logical_cpus)],["采样窗口时长",duration(r.wall_ms)],["UI 客户端数",presentNumber(r.ui_clients)],["UI 元素数",presentNumber(r.ui_elements)],["浏览器探测失败",presentNumber(window?.browser_probe_failures)]]);
    table("performance-metrics",Object.entries(window?.metrics || {}).map(([name,m])=>[`${performanceLabels[name] || name} · ${name}`,presentNumber(m.count),duration(m.avg_ms),duration(m.max_ms),presentNumber(m.over_200ms)]),5,"所选窗口没有 UI 指标采样");
    table("gc-metrics",Object.entries(r.gc || {}).map(([gen,m])=>[gen,presentNumber(m.count),duration(m.avg_ms),duration(m.max_ms),presentNumber(m.over_200ms),presentNumber(m.collected),presentNumber(m.uncollectable)]),7,"所选窗口没有 GC 记录");
  }
  function renderTiming() {
    if(!snapshot)return;
    const metric=snapshot.timings[$("timing-metric").value];
    chart("timing-chart",metric?.distribution_available?metric.buckets:[]);
  }
  function renderDiagnostics(reset = false) {
    if(!snapshot)return;
    if(reset)detailPage=0;
    const cutoff=selectedRest().cutoff_hour.slice(0,13),last=snapshot.generated_at.slice(0,13);
    const source=$("detail-source").value.trim().toLowerCase();
    detailRows=(snapshot.rest_records || []).filter(r=>{
      const hour=r.day+"T"+String(r.hour).padStart(2,"0");
      return hour>=cutoff && hour<=last && ($("rest-scope").value==="all" || r.endpoint!=="health")
        && (!$("detail-endpoint").value || r.endpoint===$("detail-endpoint").value)
        && (!$("detail-reason").value || (r.outcome_reason || "legacy_unknown")===$("detail-reason").value)
        && (!$("detail-status").value || String(r.status_code || "unknown")===$("detail-status").value)
        && (!$("detail-client").value || (r.client_family || "unknown")===$("detail-client").value)
        && (!source || [sourceName(r),r.source_kind,r.source_name,r.source_site].join(" ").toLowerCase().includes(source));
    }).sort((a,b)=>(b.day+String(b.hour).padStart(2,"0")).localeCompare(a.day+String(a.hour).padStart(2,"0")) || b.count-a.count);
    const pages=Math.max(1,Math.ceil(detailRows.length/50));detailPage=Math.min(detailPage,pages-1);
    const count=detailRows.reduce((total,r)=>total+r.count,0),rejected=detailRows.reduce((total,r)=>total+(r.status_code===429?r.count:0),0);
    $("detail-summary").textContent=`筛选后 ${number(count)} 次请求 · ${number(rejected)} 次 429 · ${number(detailRows.length)} 条聚合记录；每页 50 条，按小时倒序、次数降序。`;
    table("diagnostic-details",detailRows.slice(detailPage*50,(detailPage+1)*50).map(r=>[`${r.day} ${String(r.hour).padStart(2,"0")}:00`,r.endpoint,sourceName(r),r.client_family || "unknown",`${r.status_code || "unknown"} · ${reasonName(r.outcome_reason || "legacy_unknown")} (${r.outcome_reason || "legacy_unknown"})`,number(r.count),duration(r.count?r.sum_ms/r.count:null),latencyBucket(r.latency_bucket),`${r.limit_bucket || "—"} / ${r.top_k_bucket || "—"}`,r.parameter_bucket || "—",`${presentNumber(r.peak_in_flight)} / ${presentNumber(r.peak_per_minute)}`]),11,"没有匹配的聚合记录，请调整筛选或时间范围");
    $("detail-page").textContent=`${detailPage+1} / ${pages} 页`;$("detail-prev").disabled=detailPage===0;$("detail-next").disabled=detailPage>=pages-1;
  }
  function renderRest() {
    if(!snapshot)return;
    const rest=selectedRest(), label=$("range").selectedOptions[0].textContent;
    document.querySelectorAll(".window-label").forEach((node)=>{node.textContent=label;});
    $("window-count").textContent=number(rest.count);
    $("window-errors").textContent=`${number(rest.statuses["4xx"])} / ${number(rest.statuses["5xx"])}`;
    $("window-slow").textContent=`${number(rest.slow_count)} · ${percent(rest.slow_percent)}`;
    $("window-status").textContent=["2xx","3xx","4xx","5xx"].map((key)=>`${key} ${number(rest.statuses[key])}`).join(" / ");
    $("window-range").textContent=rest.first_hour?`记录 ${rest.first_hour.replace("T"," ")} — ${rest.last_hour.replace("T"," ")} · ${rest.observed_hours} / ${rest.requested_hours} 个小时有记录` : "当前窗口没有 REST 观测记录。";
    const reasons=rest.outcome_reasons || {},codes=rest.status_codes || {},entries=Object.entries(reasons);
    const totalFor=(predicate)=>entries.reduce((total,[reason,count])=>total+(predicate(reason)?count:0),0);
    const rate=totalFor(reason=>reason.endsWith("_rate_limited") && reason!=="authentication_rate_limited"),concurrency=totalFor(reason=>reason.endsWith("_concurrency_limited")),quota=totalFor(reason=>reason.endsWith("_daily_quota_exhausted")),validation=totalFor(reason=>reason.startsWith("validation_"));
    const failures=entries.filter(([key])=>!["success","redirect","legacy_unknown"].includes(key)).sort((a,b)=>b[1]-a[1]);
    $("rest-insight").textContent=rest.count?`${label} · ${$("rest-scope").selectedOptions[0].textContent}：429 ${number(codes["429"])} 次（${(100*(codes["429"] || 0)/rest.count).toFixed(1)}%），5xx ${number(rest.statuses["5xx"])} 次。${failures.length?`主要非成功原因：${reasonName(failures[0][0])} ${number(failures[0][1])} 次。`:"没有已知非成功原因。"} ${codes.unknown?`另有 ${number(codes.unknown)} 次未采集精确状态码。`:""}`:"当前窗口没有记录；时间空白不等于零流量。";
    signals("rejection-stats",[["429 拒绝",number(codes["429"])],["频率限制",number(rate)],["并发限制",number(concurrency)],["日额度不足",number(quota)],["认证前置限流",number(reasons.authentication_rate_limited)],["参数校验错误",number(validation)]]);
    ranks("status-codes",Object.entries(codes).sort((a,b)=>b[1]-a[1]).map(([key,count])=>({label:key==="unknown"?"unknown · 精确码未采集":key,count})),rest.count);
    ranks("outcome-reasons",entries.sort((a,b)=>b[1]-a[1]).map(([key,count])=>({label:`${reasonName(key)} · ${key}`,count})),rest.count);
    table("endpoints",rest.endpoints.map(row=>[`/api/${row.endpoint}${row.endpoint==="health"?"（健康检查）":""}`,number(row.count),duration(row.average_ms),p95(row),`${duration(row.success_latency.average_ms)}（${number(row.success_latency.count)} 次）`,p95(row.success_latency),percent(row.slow_percent),number(row.status_codes["429"]),number(row.client_errors),number(row.server_errors)]),10);
    ranks("sources",rest.sources,rest.count);ranks("clients",rest.clients,rest.count);
    signals("pressure-stats",[["HTTP 在途峰值",rest.peak_in_flight == null?"—":number(rest.peak_in_flight)],["分组分钟请求峰值",rest.peak_per_minute == null?"—":number(rest.peak_per_minute)],["有记录的小时",`${rest.observed_hours} / ${rest.requested_hours}`]]);
    const hourly=rest.hours.map(item=>{
      const statuses=item.status_codes || {},outcomes=item.outcome_reasons || {};
      const rejected=statuses["429"] || 0,client=(item.statuses?.["4xx"] || 0)-rejected,server=item.statuses?.["5xx"] || 0;
      const main=Object.entries(outcomes).filter(([key])=>!["success","redirect","legacy_unknown"].includes(key)).sort((a,b)=>b[1]-a[1])[0];
      return {...item,rejected,client,server,main,errors:item.count==null?null:rejected+client+server};
    });
    chart("error-trend",hourly.map(item=>({count:item.errors,stacks:{rejected:item.rejected,"client-error":item.client,"server-error":item.server},label:item.hour.replace("T"," "),short:item.hour.slice(5,10)+" "+item.hour.slice(11,13)+"时"})),true);
    table("hourly-details",hourly.slice().reverse().map(item=>[item.hour.replace("T"," "),presentNumber(item.count),item.count==null?"—":number(item.rejected),item.count==null?"—":number(item.client),item.count==null?"—":number(item.server),item.main?`${reasonName(item.main[0])} · ${number(item.main[1])}`:item.count==null?"无可用记录":item.outcome_reasons?.legacy_unknown?"历史原因未采集":"—",presentNumber(item.peak_in_flight),presentNumber(item.peak_per_minute)]),8);
    renderDiagnostics();
    trend();
  }
  function render(data) {
    snapshot = data;
    const c = data.counters, sum = (names) => names.reduce((total,key) => total + (c[key] || 0),0);
    $("updated").textContent = `页面快照生成 ${date(data.generated_at)}`;
    $("data-boundary").textContent=`累计事件自 ${date(data.telemetry_since)} 起；REST 自 ${date(data.attribution_since)} 采集，最后有记录的小时 ${data.rest.last_hour?.replace("T"," ") || "—"}。页面快照生成时间不等于 OSS 同步时间。${data.attribution_updated_at?`此回放的 REST 持久化快照更新于 ${date(data.attribution_updated_at)}。`:"当前读取服务内存，不据此判断 OSS 是否已同步。"}${data.telemetry_updated_at?`累计持久化快照更新于 ${date(data.telemetry_updated_at)}。`:""}小时聚合包含当前未结束小时；没有记录不等于零流量。`;
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
    chart("latency-chart", data.ui_latency.distribution_available?data.ui_latency.buckets:[]);
    $("feedback-summary").textContent=`内存保留 ${number(data.feedback_count)} 条反馈记录（受保留上限影响），累计反馈事件见下表。`;
    table("counter-details",Object.entries(c).sort((a,b)=>a[0].localeCompare(b[0])).map(([key,count])=>[eventLabels[key] || key,key,number(count)]),3);
    table("timing-details",Object.entries(data.timings).map(([key,value])=>[timingLabels[key] || key,number(value.count),duration(value.average_ms),p95(value),value.distribution_available?"是":"否"]),5);
    options("timing-metric",Object.keys(data.timings),null,key=>timingLabels[key] || key);renderTiming();
    const records=data.rest_records || [],unique=(key,fallback)=>[...new Set(records.map(row=>String(row[key] || fallback || "")))].filter(Boolean).sort();
    options("detail-endpoint",unique("endpoint"),"全部接口");options("detail-reason",unique("outcome_reason","legacy_unknown"),"全部原因",value=>`${reasonName(value)} · ${value}`);options("detail-status",unique("status_code","unknown"),"全部状态码");options("detail-client",unique("client_family","unknown"),"全部客户端");
    const q=data.quality;
    signals("quality-stats",[["发生选词的搜索会话",percent(q.selection_percent)],["复制操作 / 搜索",percent(q.copy_events_per_search)],["零结果率",percent(q.zero_percent)],["60 秒内重复搜索",percent(q.repeat_percent)],["冷启动成功 / 尝试",`${number(q.cold_successes)} / ${number(q.cold_attempts)}`],["冷启动失败",number(q.cold_failures)]]);
    renderHistory();renderRest();
  }
  async function refresh() {
    if (loading || document.hidden) return;
    loading = true; $("refresh").disabled = true;
    const viewAtStart=keysPage;
    try {
      if (!await session()) return;
      if(viewAtStart!==keysPage)return;
      if (keysPage) {await window.keyPortal?.refresh();return;}
      const response = await request("/admin/api/overview", {cache:"no-store",credentials:"same-origin"});
      if (response.status === 401 || response.status === 403) {clearPrivateView();notice("后台登录已失效，请重新登录。");await session();return;}
      if (!response.ok) throw new Error("metrics");
      render(await response.json()); notice("");
    } catch (_) {notice(snapshot ? "刷新失败，当前保留上一次快照。请稍后重试，并注意快照时间。" : "暂时无法读取后台状态，请稍后刷新。标签搜索可从左侧入口访问。");}
    finally {loading=false;$("refresh").disabled=false;if(viewAtStart!==keysPage)refresh();}
  }
  $("refresh").addEventListener("click", refresh);
  for(const id of ["range","rest-scope"])$(id).addEventListener("change",()=>{detailPage=0;renderRest();});
  $("performance-window").addEventListener("change",renderPerformanceDetails);
  $("timing-metric").addEventListener("change",renderTiming);
  for(const id of ["detail-endpoint","detail-reason","detail-status","detail-client"])$(id).addEventListener("change",()=>renderDiagnostics(true));
  $("detail-source").addEventListener("input",()=>renderDiagnostics(true));
  $("detail-reset").addEventListener("click",()=>{for(const id of ["detail-endpoint","detail-reason","detail-status","detail-client","detail-source"])$(id).value="";renderDiagnostics(true);});
  $("detail-prev").addEventListener("click",()=>{detailPage--;renderDiagnostics();});
  $("detail-next").addEventListener("click",()=>{detailPage++;renderDiagnostics();});
  $("logout").addEventListener("click", async () => {
    $("logout").disabled=true;
    try {
      const response = await request("/admin/logout", {method:"POST",credentials:"same-origin",headers:{"X-CSRF-Token":csrf || ""}});
      if (!response.ok && response.status !== 401) throw new Error("logout");
      clearPrivateView(); location.replace("/admin?notice=logged_out");
    } catch (_) {notice("退出未完成，请重试。当前登录状态尚未确认失效。");}
    finally {$("logout").disabled=false;}
  });
  function selectView() {
    keysPage = location.pathname.replace(/\/$/, "") === "/admin/api-keys";
    document.querySelectorAll("[data-nav]").forEach(link => {
      if(link.dataset.nav === (keysPage ? "keys" : "overview")) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    $("overview").hidden = keysPage; $("keys").hidden = !keysPage;
    refresh();
  }
  document.querySelectorAll("[data-nav]").forEach(link => link.addEventListener("click", event => {
    if(event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
    event.preventDefault();history.pushState(null, "", link.getAttribute("href"));selectView();
  }));
  window.addEventListener("popstate", selectView);
  const messages={forbidden:"HF 身份验证成功，但该账号没有后台管理员权限。请使用指定管理员账号登录。",login_failed:"登录验证未完成或已过期，请重新发起登录。",busy:"登录请求较多，请几分钟后重试。",unconfigured:"后台 OAuth 尚未配置，暂时不能登录。",logged_out:"已退出后台。"};
  const message=messages[new URLSearchParams(location.search).get("notice")];if(message) notice(message);
  // Avoid displaying previous private DOM when restoring a browser history entry.
  window.addEventListener("pagehide", clearPrivateView);
  window.addEventListener("pageshow", (event) => {if(event.persisted) refresh();});
  document.addEventListener("visibilitychange", () => {if(!document.hidden) refresh();});
  setInterval(refresh, 30000);
  let resizeTimer;
  window.addEventListener("resize", () => {clearTimeout(resizeTimer);resizeTimer=setTimeout(() => {if(snapshot && !keysPage) {chart("latency-chart",snapshot.ui_latency.distribution_available?snapshot.ui_latency.buckets:[]);renderTiming();renderHistory();renderRest();}},100);});
  refresh();
})();

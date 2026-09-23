"use strict";
(() => {
  const $ = id => document.getElementById(id), admin = location.pathname.startsWith("/admin/");
  let csrf = null, terms = "", page = 0, submission = crypto.randomUUID(), target = null, expiry;
  const errors = {pending_application_exists:"已有待审核申请，请等待审核。", review_reason_required:"请补充申请原因；第二把 Key 必须人工审核。", stale_version_refresh_required:"记录已变更，请刷新后重试。", key_service_unconfigured:"服务尚未配置完成，请联系维护者。", key_service_unavailable:"额度服务暂不可用，请稍后重试。", already_claimed_use_rotate:"已领取；若未保存，请轮换 Key。", invalid_application_fields:"请检查申请字段、Client 格式及 HTTPS 地址。", login_required:"登录已过期，请重新登录。"};
  function message(text) { $("message").textContent = text; }
  async function request(path, data) {
    const r = await fetch(path, {method:data === undefined?"GET":"POST", cache:"no-store", credentials:"same-origin", headers:data === undefined?{}:{"Content-Type":"application/json","X-CSRF-Token":csrf || ""}, body:data === undefined?undefined:JSON.stringify(data)});
    const result = await r.json();
    if (!r.ok) { if (r.status===401) clear(); throw new Error(errors[result.error] || result.error || result.detail || "操作失败，请刷新后重试。"); }
    return result;
  }
  function clearSecret(){ $("secret").value=""; if($("secret-dialog").open) $("secret-dialog").close(); }
  function clear(){csrf=null;clearTimeout(expiry);clearSecret();$("content").hidden=true;$("logout").hidden=true;$("login").hidden=false;for(const id of ["applications","grants","audit"]) $(id).replaceChildren();}
  function button(root,text,action){const b=document.createElement("button");b.textContent=text;b.onclick=async()=>{b.disabled=true;try{await action();}catch(e){message(e.message);}finally{b.disabled=false;}};root.append(b);}
  function line(root,text){const p=document.createElement("p");p.textContent=text;root.append(p);}
  function card(root,title){const c=document.createElement("article");c.className="card";const h=document.createElement("h3");h.textContent=title;c.append(h);root.append(c);return c;}
  const states={pending:"待审核",approved:"已批准",rejected:"已拒绝",active:"有效",paused:"已暂停",closed:"已关闭"};
  async function adminAction(data){await request("/admin/api/key-action",data);message("操作已保存。");await load();}
  async function grantAction(g,action){
    if(!confirm(action==="revoke"?"永久吊销这把逻辑 Key？旧凭证无法恢复。":action==="rotate"?"生成新 Key 并立即撤销旧 Key？额度不会重置。":"领取 Key？完整值只展示一次。"))return;
    const result=await request(`/developer/api/grants/${g.id}/${action}`,{version:g.version});
    if(result.key){$("secret").value=result.key;$("secret-dialog").showModal();}
    await load();
  }
  function edit(g){target=g.id;submission=crypto.randomUUID();const form=$("apply-form");for(const key of ["kind","client","site","daily"])form.elements[key].value=g[key];$("apply-title").textContent="申请增额 / 变更业务资料（人工审核）";$("cancel-edit").hidden=false;form.elements.reason.required=true;$("apply-section").scrollIntoView();estimate();}
  async function load(){
    const query=new URLSearchParams({page});if(admin){query.set("search",$("filter-search").value);query.set("state",$("filter-state").value);query.set("grant_state",$("filter-grant").value);}
    const data=await request((admin?"/admin/api/key-data":"/developer/api/data")+"?"+query);
    $("page").textContent=`第 ${page+1} 页 · 每类最多 25 条`;$("previous").disabled=page===0;
    $("reset").textContent="下次日额度重置："+new Date(data.reset_at).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai"})+"（北京时间）";
    $("applications").replaceChildren();$("grants").replaceChildren();
    for(const a of data.applications){
      const c=card($("applications"),`${a.client} · ${states[a.state]}`);
      line(c,`${a.kind==="personal"?"个人":"公开"}业务 · 申请 ${a.daily} 点/日${a.target_grant?" · 变更已有授权":""}`);
      line(c,`账号：${a.username} · 申请编号：${a.id}`);if(a.site)line(c,"Site："+a.site);line(c,"用途："+a.purpose);if(a.reason)line(c,"申请原因："+a.reason);if(a.review_reason)line(c,"审核说明："+a.review_reason);
      if(admin && a.state==="pending")for(const decision of ["approve","reject"])button(c,decision==="approve"?"批准":"拒绝",async()=>{
        const reason=prompt("填写用户可见审核说明");if(!reason)return;
        const daily=decision==="approve"?Number(prompt("核定每日点数（个人最高 6000）",a.daily)):a.daily;
        const internal_note=prompt("内部备注（可留空，用户不可见）") || "";
        await adminAction({action:"review",id:a.id,decision,daily,reason,internal_note});
      });
    }
    for(const g of data.grants){
      const c=card($("grants"),`${g.client} · ${states[g.state]}${g.state==="active"&&!g.key_id?" · 待领取 / 重新生成":""}`);
      line(c,`${g.kind==="personal"?"个人":"公开"}业务 · ${g.username} · 今日 ${g.used} / ${g.daily} 点 · 剩余 ${g.remaining}`);
      line(c,`Client：${g.client}${g.site?"\nSite："+g.site:""}`);line(c,`前缀：${g.prefix || "无有效凭证"}\n授权编号：${g.id}\n最后受理：${g.last_used_at || "尚无"}`);
      if(admin){
        if(g.state!=="closed"){
          button(c,"调整额度 / 暂停恢复",async()=>{const daily=Number(prompt("每日点数",g.daily));const state=prompt("授权状态：active 或 paused",g.state);const reason=prompt("变更原因");if(reason)await adminAction({action:"update",grant_id:g.id,version:g.version,daily,state,reason});});
          button(c,"永久吊销",async()=>{const reason=prompt("确认永久吊销，填写原因");if(reason)await adminAction({action:"revoke",grant_id:g.id,version:g.version,reason});});
        }else button(c,"重新批准授权",async()=>{const reason=prompt("确认重新批准，旧秘密不恢复；填写原因");if(reason)await adminAction({action:"reopen",grant_id:g.id,version:g.version,reason});});
      }else{
        if(g.state==="active"){
          if(!g.key_id)button(c,"首次领取",()=>grantAction(g,"claim"));
          button(c,"轮换 / 重新生成",()=>grantAction(g,"rotate"));
        }
        if(g.state!=="closed"){button(c,"增额 / 变更资料",()=>edit(g));button(c,"吊销",()=>grantAction(g,"revoke"));}
      }
    }
    if(!data.applications.length)line($("applications"),"暂无申请记录。");if(!data.grants.length)line($("grants"),"暂无授权。");
    if(admin){
      const used=(data.pool.rows.find(x=>x.subject==="anonymous")||{}).used||0;
      $("pool").textContent=`模式：${data.mode} · 匿名池 ${used}/${data.anonymous_daily} 点，剩余 ${Math.max(0,data.anonymous_daily-used)} · 匿名并发 ${data.in_flight.anonymous||0}/1 · 测试池并发 ${data.in_flight.anonymous_preview||0}/1 · 短时拒绝 ${data.rejections} · 待退 ${data.pool.pending_refunds}，本进程待退 ${data.local_pending_refunds}`;
      $("audit").replaceChildren();for(const row of data.audit.rows){const c=card($("audit"),`${row.action} · ${row.actor}`);line(c,`${row.created_at} · ${row.target} · ${row.reason}`);line(c,JSON.stringify({before:row.before_value,after:row.after_value}));}
    }
  }
  async function session(){
    clear();const path=admin?"/admin/api/session":"/developer/api/session";
    const r=await fetch(path,{cache:"no-store",credentials:"same-origin"}), data=await r.json();
    if(!r.ok||!data.authenticated){$("login-link").hidden=!data.login_url;if(data.login_url)$("login-link").href=data.login_url;message(location.search.includes("login_failed")?"登录未完成或账号不在测试白名单，请使用获准的 HF 账号。":"请先登录。测试期间仅受邀账号可以申请。");return;}
    csrf=data.csrf;terms=data.terms || "";$("login").hidden=true;$("content").hidden=false;$("logout").hidden=false;
    $("identity").textContent=data.username+(data.mode==="preview"?" · 测试模式，普通用户调用不受影响":"");
    expiry=setTimeout(()=>{clear();message("登录已过期，请重新登录。");},data.expires_in*1000);
    await load();message(data.mode==="preview"?"当前仅为受邀测试。正式开放由维护者手动开启。":"");
  }
  function estimate(){const daily=Number($("daily").value);$("estimate").textContent=`约相当于每天 ${Math.floor(daily/3)} 次纯搜索，或 ${Math.floor(daily/2)} 次纯关联，或 ${daily} 次纯画师推荐。`;$ ("site").required=$("kind").value==="public";}
  $("kind").onchange=()=>{$("daily").value=$("kind").value==="public"?10000:3000;estimate();};$("daily").oninput=estimate;
  $("apply-form").onsubmit=async e=>{e.preventDefault();const form=e.currentTarget,b=form.querySelector('[type="submit"]');b.disabled=true;try{const raw=Object.fromEntries(new FormData(form));await request("/developer/api/apply",{...raw,daily:Number(raw.daily),accepted:raw.accepted==="on",terms,submission_id:submission,grant_id:target});submission=crypto.randomUUID();message("申请已提交，请查看审批结果。");await load();}catch(error){message(error.message);}finally{b.disabled=false;}};
  $("cancel-edit").onclick=()=>{target=null;submission=crypto.randomUUID();$("apply-form").reset();$("apply-form").elements.reason.required=false;$("apply-title").textContent="申请独立额度";$("cancel-edit").hidden=true;estimate();};
  for(const id of ["refresh-login","refresh"])$(id).onclick=()=>session().catch(e=>message(e.message));
  $("logout").onclick=async()=>{try{await request(admin?"/admin/logout":"/developer/logout",{});clear();message("已退出。");}catch(e){message(e.message);}};
  $("previous").onclick=()=>{page=Math.max(0,page-1);load().catch(e=>message(e.message));};$("next").onclick=()=>{page++;load().catch(e=>message(e.message));};
  $("filter").onclick=()=>{page=0;load().catch(e=>message(e.message));};$("retry-refunds").onclick=()=>adminAction({action:"retry_refunds"}).catch(e=>message(e.message));
  $("close-secret").onclick=clearSecret;$("secret-dialog").addEventListener("close",()=>{$("secret").value="";});$("copy-secret").onclick=()=>navigator.clipboard.writeText($("secret").value).catch(()=>message("自动复制不可用，请手动复制。"));
  window.addEventListener("pagehide",clear);window.addEventListener("pageshow",e=>{if(e.persisted)session().catch(err=>message(err.message));});
  $("admin-link").hidden=!admin;$("admin-tools").hidden=!admin;$("audit-section").hidden=!admin;$("apply-section").hidden=admin;$("rules").hidden=admin;$("title").textContent=admin?"API Key 审核与管理":"API 接入";
  estimate();session().catch(e=>message(e.message));
})();

/* Pure display helpers shared by the portal and local unit tests. */
((root) => {
  const states = {active:"有效",paused:"已暂停",closed:"已关闭"};
  const kinds = {personal:"个人业务",public:"公开业务"};
  function sections(path, editing=false) {
    const apply = path.replace(/\/$/, "") === "/developer/apply" || editing;
    return {apply, applications:apply, grants:!apply};
  }
  function audit(row, names={}) {
    const before=row.before_value || {}, after=row.after_value || {};
    const actions={"configuration:update":"更新部署额度配置",apply:"提交申请","auto_approve:v1":"系统自动批准","review:approve":"批准申请","review:reject":"拒绝申请",claim:"领取 Key",rotate:"轮换 Key",revoke:"吊销 Key",reopen:"重新批准授权",update:"调整授权"};
    const notes={"configuration:update":"默认申请额度或匿名池日额度已调整；已有授权和用量保留。",apply:"申请已提交，等待审核。","auto_approve:v1":"符合个人首把 Key 的自动批准条件。","review:approve":"申请已批准。","review:reject":"申请未通过审核。",claim:"用户已领取 Key，完整凭证仅展示一次。",rotate:"已生成新 Key，旧 Key 立即失效；用量不重置。",revoke:"授权已关闭，原 Key 已失效。",reopen:"授权已重新开放，用户需重新生成 Key；旧 Key 不会恢复。"};
    const changes=[];
    for(const [field,label,format] of [
      ["personal_daily","个人默认日额度",v=>`${v} 点`],["public_daily","公开建议日额度",v=>`${v} 点`],["anonymous_daily","匿名池日额度",v=>`${v} 点`],
      ["daily","每日额度",v=>`${Number(v).toLocaleString("zh-CN")} 点`],
      ["state","授权状态",v=>states[v] || "未知状态"],
      ["kind","业务类型",v=>kinds[v] || "未知类型"],
      ["client","Client",String],["site","服务地址",v=>v || "未登记"]
    ]) {
      if(after[field] !== undefined && before[field] !== after[field])
        changes.push(`${label}：${before[field] === undefined ? "" : format(before[field])+" → "}${format(after[field])}`);
    }
    const time=new Date(row.created_at);
    const title=row.action==="update" && before.state!==after.state && after.state
      ? (after.state==="paused"?"暂停授权":after.state==="active"?"恢复授权":"调整授权")
      : actions[row.action] || "其他管理操作";
    return {title, client:after.client || before.client || "",
      actor:row.action.startsWith("auto_approve:") || row.actor.startsWith("system:") ? "系统" : names[row.actor] || `HF 账号（${row.actor.slice(0,8)}…）`,
      time:Number.isNaN(time.getTime()) ? "时间未知" : time.toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false})+"（北京时间）",
      description:notes[row.action] || "授权设置已更新。",changes,reason:row.reason || ""};
  }
  function searchExamples(key, grant) {
    const headers=["Content-Type: application/json", "Authorization: Bearer "+key,
      "X-DanbooruSearch-Client: "+grant.client];
    if(grant.kind==="public") headers.push("X-DanbooruSearch-Site: "+grant.site);
    const url="https://sakizuki-danboorusearch.hf.space/api/search";
    const body=JSON.stringify({query:"白色水手服",limit:5});
    const ps=value=>"'"+value.replace(/'/g,"''")+"'";
    const sh=value=>"'"+value.replace(/'/g,"'\"'\"'")+"'";
    return {
      // stdin avoids native JSON argument quoting differences across PowerShell 7 versions.
      windows:["& {", "  $OutputEncoding = [System.Text.UTF8Encoding]::new()",
        "  "+ps(body)+" | curl.exe -sS -i "+ps(url)+" `",
        ...headers.map(header=>"    -H "+ps(header)+" `"), "    --data-binary '@-'", "}"].join("\n"),
      posix:["curl -sS -i "+sh(url),...headers.map(header=>"  -H "+sh(header)),
        "  --data-binary "+sh(body)].join(" \\\n"),
      python:["import json", "import urllib.request", "import urllib.error", "",
        "headers = "+JSON.stringify(Object.fromEntries(headers.map(header=>{
          const split=header.indexOf(": ");return [header.slice(0,split),header.slice(split+2)];
        })),null,4),
        'data = json.dumps({"query": "白色水手服", "limit": 5}, ensure_ascii=False).encode("utf-8")',
        "request = urllib.request.Request("+JSON.stringify(url)+", data=data, headers=headers)",
        "try:","    response = urllib.request.urlopen(request, timeout=30)",
        "except urllib.error.HTTPError as error:","    response = error",
        "with response:","    print(response.status)", '    print(response.read().decode("utf-8"))'].join("\n")
    };
  }
  const api={sections,audit,searchExamples};
  if(typeof module!=="undefined" && module.exports) module.exports=api;
  else root.KeyPortalView=api;
})(globalThis);

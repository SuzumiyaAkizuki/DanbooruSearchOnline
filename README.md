---
title: DanbooruSearch
emoji: 👁
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 基于语义匹配的 Danbooru 标签搜索引擎，支持多维匹配、智能分词与共现关联推荐。
tags:
  - text-to-image
  - prompt-engineering
  - stable-diffusion
  - danbooru
  - nlp
thumbnail: >-
    https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022117025.png
---

# Danbooru Tags Searcher

![](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022117025.png)

本项目提供一种基于语义匹配的 Danbooru 标签查找方案。用户可以通过输入日常语言或模糊描述，查找并匹配 Danbooru 数据集中对应的标准标签，适用于辅助构建 Stable Diffusion、NovelAI 等 AI 绘画工具的提示词（Prompt）。

目前支持使用汉语和英语进行查找

**立即使用：** https://huggingface.co/spaces/SAkizuki/DanbooruSearch

**ComfyUI 插件版本：** [ComfyUI-DanbooruSearcher](https://github.com/SuzumiyaAkizuki/ComfyUI-DanbooruSearcher)

本项目的数据库由Danbooru API抓取而成，使用LLM进行语义扩充和中文翻译。角色名、作品名使用二次元垂类数据库Bangumi API进行精确查询，尽可能地避免了在角色名、作品名上的幻觉问题。

**数据库生成代码：** https://github.com/SuzumiyaAkizuki/danbooru-tag-pipeline

---

## 核心特性

本版本在基础语义检索之上，引入了完整的标签工作流辅助功能：

* **多维度语义匹配：** 底层支持 4 个维度的向量空间检索（英文、中文扩展词、维基释义、中文核心词），大幅提升模糊描述的命中率。
* **智能分词与显式分隔：** 支持一键开启/关闭自动分词。开启后自动提取长句中概念分别检索；同时支持用空格、换行、中文逗号（，）和顿号（、）手动分隔概念，系统会尊重你的分隔意图，将每个片段作为整体匹配（超过 7 字的片段仍然自动分词）。
* **标签互相关推荐：** 引擎内置标签共现数据。当你选中某些标签后，系统会自动推荐在 Danbooru 图库中经常与它们一同出现的相关标签，辅助补全画面细节。
* **推荐擅长画师：** 基于标签-画师 NPMI 共现数据，根据已选标签实时推荐擅长绘制这些元素的画师，支持勾选加入已选标签。
* **精细化查阅与过滤：** 支持按标签类别（General通用 / Character角色 / Copyright作品）进行硬过滤。
  * 提供 **NSFW 保护模式**（开启后自动模糊敏感词汇）。
  * 表格内置 Danbooru 维基释义悬浮提示，并提供前往原站的快捷链接。


## 快速了解

### 功能概览

本工具提供以下核心功能：

- [精确查词](#1-精确查词)（容错拼写错误）
- [概念模糊发散](#2-概念模糊发散)
- [输入描述查词](#3-输入描述查词)
- [完整画面查找](#4-完整画面查找)
- [关联推荐](#5-关联推荐共现匹配)
- [推荐擅长画师](#6-推荐擅长画师)

前四项基于**语义向量匹配**，后两项基于**共现数据**（标签-标签共现与标签-画师共现）。

---

### 参数说明

在进入功能介绍之前，先了解几个关键参数，它们会影响所有搜索模式的结果质量。

**Top K（语义相关）**
针对单个分词，检索语义最相关的候选标签数量。该值作用于每个向量检索层，最终候选池大小约为 Top K × 层数，再截断至结果上限。值越大，召回越广，但热度高的分词会占据更多名额，可能挤压低频但精确的分词。精确查词场景建议调低（20），发散探索场景可以调高（80~160）。

**结果上限**
最终展示的标签总条数。建议设为 70~80 以适配主流 SDXL 模型的 prompt 长度。精确查词时可以调低到 10，避免干扰。

**热度权重（0~1）**
控制标签在 Danbooru 站内发帖量对最终排序的影响。数值越高，越倾向于推荐常用高频标签。默认值 0.15 在大多数场景下表现良好。

**智能分词**
开启时，程序会自动拆分输入的长句，提取其中的关键概念分别检索，再合并结果。适合完整画面描述；若只想检索一个完整短语的语义，建议关闭。

**新版支持显式分隔符：** 用空格、换行、或中文逗号/顿号（，、）手动分隔概念时，系统会将每个分隔片段视为独立的原子概念，不再做进一步拆分。这在你明确知道自己想要哪些概念时比自动分词更精准。

| 输入示例 | 分词结果 | 说明 |
|---|---|---|
| `运动社团 校队 比赛` | `运动社团` `校队` `比赛` | 空格分隔，每个片段原样保留 |
| `运动社团、校队、比赛` | `运动社团` `校队` `比赛` | 逗号/顿号同理 |
| `一个穿着白色水手服的少女` | `穿着` `白色` `水手服` `少女` | 无显式分隔，自动分词（停用词已过滤） |

> 当分隔后的某个片段超过 **7 个字**时，系统会退回到自动分词——因为超过这个长度的片段通常是短句而非单一概念。这一阈值基于对所有标签核心中文名长度的统计分析得出。

---

## 功能介绍

### 1. 精确查词

**适用场景：** 你想用中文或英文查询一个单一概念，或者知道某个标签的大概写法但不确定拼写是否准确。

本工具通过语义相似度匹配而非字符串精确匹配，因此具备一定的拼写容错能力。
中文概念同样会通过中文核心词、中文扩展词和释义等向量层检索，例如输入 `水手服` 可以召回 `serafuku`。语义搜索召回废弃标签时，系统会使用官方 Danbooru Tag Alias 将其规范化为当前标签；Alias 只纠正结果，不替代语义搜索。

**建议参数：** Top K: 20 | 结果上限: 10 | 热度权重: 0.15 | **关闭智能分词**

> 关闭智能分词，确保整个输入作为一个整体进行语义匹配，不会被拆散。

#### 示例

| 你的输入 | 实际标准标签 | 说明 |
|---|---|---|
| `水手服` | `serafuku` | 中文概念通过语义搜索命中 |
| `selafuku` | `serafuku` | 经典拼写错误，仍可命中 |
| `school unifor` | `school_uniform` | 漏字母，仍可命中 |
| `twintail` | `twintails` | 单复数混淆，仍可命中 |
| `zettai ryouiki` | `zettai_ryouiki` | 罗马音输入，可命中 |
| `thighhigh` | `thighhighs` | 词形变体，仍可命中 |

---

### 2. 概念模糊发散

**适用场景：** 你脑中有一个模糊的概念或风格意象，但不知道在 Danbooru 里对应哪些标签，希望系统帮你"发散"出相关的候选标签。

**建议参数：** Top K: 80~160 | 结果上限: 80 | 热度权重: 0.15 | **开启智能分词**

#### 示例

**示例 1：「兔耳朵」**

输入一个宽泛概念，系统会返回各类耳型相关标签，包括真兔耳、头饰耳、兽娘耳等变体。

![image-20260402202537091](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022025661.png)

---

**示例 2：「中国风古装」**

返回汉服、旗袍、各朝代服制等细分标签，帮助你快速了解这一风格下的标签体系。

![image-20260402202652780](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022026925.png)

---

**示例 3：「机械义肢」**

返回包含义肢、机械臂、赛博朋克肢体等标签，适合进一步结合关联推荐做细化筛选。

![image-20260402202730735](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022027911.png)

---

**示例 4：「战斗姿势」**

返回各类战斗动作、持武器动作等标签。

![image-20260402202757757](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022028668.png)

---

### 3. 输入描述查词

**适用场景：** 你能描述某个具体的事物、角色、场景，但不知道 Danbooru 的对应标签。可以使用中文或英文自然语言描述，也可以通过 IP 名称、角色外号来查找角色标签。

**建议参数：** Top K: 20 | 结果上限: 20 | 热度权重: 0.15 | **关闭智能分词**

> 关闭智能分词，使系统对你的整段描述做整体语义匹配，而不是拆词。

#### 示例

**示例 1：用描述定位角色**

| 你的输入 | 期望命中的角色标签 | 说明 |
|---|---|---|
| `原神里面须弥主线的NPC贵族女孩` | `dunyarzad_(genshin_impact)`（迪纳泽黛） | 通过剧情身份描述定位角色 |
| `命运石之门的助手` | `makise_kurisu`（牧瀬红莉栖） | 通过人物关系描述定位角色 |
| `某科学的超电磁炮中会传送的那个人` | `shirai_kuroko`（白井黑子） | 通过能力和动作描述定位角色 |
| `EVA 里驾驶零号机的蓝发角色` | `ayanami_rei`（凌波丽） | 通过机体和外貌描述定位角色 |
| `明日方舟里游戏商店的老板` | `closure_(arknights)`（可露希尔） | 通过职位描述定位角色 |

---

**示例 2：用描述定位服饰/物品**

| 你的输入 | 期望命中的标签 | 说明 |
|---|---|---|
| `紧身衣勒入皮肤产生的凹陷` | `skindentation` | 非常具体的视觉细节描述 |
| `半透明薄纱覆盖胸前` | `see-through_bra` | 材质与遮盖关系描述 |
| `两侧有开缝、有拉绳的运动短裤` | `dolphin_shorts` | 服饰细节描述 |

---

### 4. 完整画面查找

**适用场景：** 你脑中已经有了一幅完整的画面，希望一次性把整段描述转换成尽可能多的 Danbooru 标签，用于 AI 绘画的完整 Prompt。

**建议参数：** Top K: 5 | 结果上限: 80 | 热度权重: 0.15 | **开启智能分词**

> 开启智能分词：系统会自动拆分你的长句，提取其中的关键概念分别检索，再合并去重，覆盖尽可能多的画面元素。

#### 示例

**示例 1：雨中奔跑的少女**

```
一个穿着白色水手服、蓝色短裙的少女，在下着大雨的城市街道奔跑，
她的表情是不甘、愤怒、流泪，她的衣服湿透。
```

> short_dress, city, streaming_tears, street, white_serafuku, furious, fume, rain, tears, running, skirt, crying, tearing_up, urban

**示例 2：赛博朋克城市夜景**

```
赛博朋克风格的城市，霓虹灯招牌，雨后潮湿的街道倒映灯光，
远处高楼林立，一个穿着皮夹克的女孩站在路灯下抽烟。
```

> street, reflection, lights, lamp, leather_jacket, fur_jacket, 1girl, standing, lamppost, smoking, holding_cigarette, cyberpunk, after_rain, city, urban, neon_lights, wet_floor, utility_pole

---

**示例 3：奇幻战场**

```
一个银发的精灵女战士，身穿轻甲，手持弓箭，站在废墟上，
背景是正在燃烧的城市和布满乌云的天空，衣服上有血迹。
```

> burning, city, dark_cloud, sky, clothes, blood_stain, blood_on_clothes, silver_hairband, standing, white_hair, cyberpunk, after_rain, holding, bow_(weapon), urban, ruins, elf, rubble, holding_bow_(weapon), arrow_(projectile), neon_lights, burning_building, wet_floor, utility_pole

---

**技巧：关于描述的写法**

- 分段描述不同要素（人物 / 背景 / 表情 / 服装）比混在一起效果更好
- 细节越具体，匹配越准确——「蓝色长发」优于「特别的发色」
- 遇到感觉应该有但没命中的标签，可以把该部分单独截取，切换到「输入描述查词」模式单独查

---

### 5. 关联推荐（共现匹配）

关联推荐与前四项功能机制不同，它不依赖语义相似度，而是基于 **Danbooru 图库中标签的共现统计**：如果两个标签经常出现在同一张图上，就认为它们相关。

**使用方式：** 在左侧结果表格中勾选你感兴趣的标签，右侧关联推荐栏会自动刷新，显示在 Danbooru 图库中与你选中标签最常共现的其他标签。也可以点击「根据已选刷新」手动触发。

关联推荐的得分是基于 NPMI（互信息归一化）计算的，能有效过滤掉那些只是因为自身热度高而频繁出现的标签。

#### 适用场景

**场景 1：查看某种服饰/道具的常见搭配**

勾选 `fingerless_gloves`（露指手套），关联推荐会列出穿露指手套的角色们。

![image-20260402204128858](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022041742.png)

---

**场景 2：查看某个角色的典型特征**

勾选角色标签（如 `amiya_(arknights)`），关联推荐会列出该角色最常出现的服装、配件、表情等标签，相当于自动生成角色的"标签画像"。

![image-20260402204239836](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022042492.png)

---

**场景 3：探索某类主题的标签体系**

勾选 `fighter_jet`（战斗机），关联推荐会列出各类机型、飞行动作、背景等相关标签，帮助你快速摸清一个主题下的标签生态。

![image-20260402204306860](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022043665.png)

---

**场景 4：组合多个标签获得交集推荐**

同时勾选多个标签时，关联推荐会综合计算多标签的共现关系，推荐与这组标签**整体**最匹配的标签，而不只是单个标签的邻居。

比如同时勾选 `maid` + `twintails`，推荐结果会更倾向于「女仆且双马尾」场景下经常出现的标签，而不是单纯「女仆」或「双马尾」各自的高频搭配。

![image-20260402204415282](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202604022044935.png)

---

### 6. 推荐擅长画师

推荐擅长画师与关联推荐类似，基于 **标签-画师 NPMI 共现数据**，但目标是推荐擅长绘制已选标签所描述元素的画师。

**使用方式：** 在左侧结果表格或右侧关联推荐中勾选标签后，右栏「推荐擅长画师」区域会**实时自动更新**，无需手动刷新。画师可勾选加入已选标签（Anima 模式下复制时自动加 `@` 前缀）。

**评分逻辑：** 对每个已选标签，计算该标签与画师的 NPMI 共现分，累加后乘以命中标签数加成，最终归一化显示为百分比。

#### 适用场景

**场景 1：根据风格标签找画师**

勾选 `flat_color`（平涂）+ `1girl`，推荐擅长平涂风格女性角色的画师。

**场景 2：根据角色找画师**

勾选某角色标签，推荐经常绘制该角色的画师。

**场景 3：根据画面元素找画师**

勾选 `mecha`（机甲）+ `detailed_background`，推荐擅长机甲和精细背景的画师。

---

### 使用技巧

**技巧 1：先宽后窄**
先用完整画面查找拿到大量候选标签，再用分词筛选 chip 或手动去掉不相关的，然后勾选剩余标签触发关联推荐，补充遗漏细节。

**技巧 2：拿不准标签含义时悬浮查看**
鼠标悬停在任意标签行或关联推荐行上，会弹出该标签的 Danbooru 维基释义，方便快速判断是否符合需求，无需跳转站外。

**技巧 3：点击标签名直接跳转**
标签名是可点击链接，点击后在新标签页打开 Danbooru 的对应维基页，可以查看详细说明、相关图例和近义标签。

**技巧 4：NSFW 保护模式**
默认关闭 NSFW 显示。打开后标签的完整名称和含义会显示出来，但平台限制下（魔搭创空间）此选项不可用。

**技巧 5：复制全部 vs 复制选中**
- **复制全部标签**：把当前搜索结果中所有标签复制为逗号分隔的字符串，适合快速生成完整 Prompt
- **复制选中标签**：只复制你在表格中手动勾选的标签，适合精选后使用

**技巧 6：用显式分隔符精细控制分词**
当你知道自己想要哪些独立概念时，用空格或中文逗号手动分隔它们：
- 「`运动社团 校队 比赛 运动会 体育祭 田径部`」→ 每个概念原样保留
- 「`反乌托邦、赛博朋克、蒸汽朋克、废土`」→ 逗号分隔同理
- 分隔后的短片段（≤7 字）作为整体匹配，不会被错误拆开
- 如果需要混合长短查询，可以一部分用空格分隔、另一部分写成自然语句

---

## 接口方法

### REST API

本工具同时提供 REST API，可集成到你自己的工作流中。

启动后访问 `/api/docs` 查看完整的交互式 API 文档。

**搜索接口**
```
POST /api/search
{
    "query": "白色水手服的女孩",
    "top_k": 5,
    "limit": 80,
    "popularity_weight": 0.15,
    "show_nsfw": true,
    "use_segmentation": true,
    "target_layers": ["英文", "中文扩展词", "释义", "中文核心词"],
    "target_categories": ["General", "Character", "Copyright"],
    "group_mode": "off",
    "max_per_group": 2
}
```

搜索接口保留底层可调参数，适合需要精细控制召回数量、分层检索、分词策略、类别筛选和分组去重的工作流。
响应包含 `tags_all`、`tags_sfw`、`keywords`、`results`；每条结果默认包含完整字段和 `wiki`。如果语义召回结果经官方 Tag Alias 规范化，还会包含原废弃标签 `alias_from`；`tag`、`tags_all` 和 `tags_sfw` 始终使用规范标签。

**关联推荐接口**
```
POST /api/related
{
    "tags": ["white_serafuku", "sailor_collar"],
    "limit": 50,
    "show_nsfw": true
}
```

响应为 `results` 列表，每条结果默认包含 `wiki`；如果输入标签拼写错误但可被纠正，会额外返回 `correction_note` 和 `corrections`。

**推荐擅长画师接口**
```
POST /api/artists
{
    "tags": ["1girl", "blue_hair", "school_uniform"],
    "limit": 30,
    "min_cooc": 3,
    "show_nsfw": true
}
```

响应字段：`artist`（画师名）、`cooc_count`（共现次数）、`post_count`（作品数）、`sources`（命中标签）、`top_tags`（该画师最常画的 10 个标签，含中文名）。

**健康检查接口**

```
GET /api/health
```

详见：[API 文档](https://sakizuki-danboorusearch.hf.space/api/docs)

---

### MCP 接口

本工具提供 MCP（Model Context Protocol）服务，支持将搜索引擎直接接入 Claude Desktop、Cursor、Cherry Studio 等支持 MCP 协议的大模型客户端，让 AI 能够在对话中直接调用标签搜索能力。

**MCP 服务地址：**

```
https://sakizuki-danboorusearch.hf.space/mcp/mcp
```

```
https://sakizuki-danboorusearchonline.ms.show/mcp/mcp
```

以上两个端口提供完全相同的服务，互为备份

#### 接入方法

**方法一：Claude Desktop**

Claude Desktop 不支持直接连接 Streamable HTTP 端点，需要通过 `mcp-remote` 作为本地桥接。

首先全局安装 `mcp-remote`（需要本机已安装 Node.js）：

```bash
npm install -g mcp-remote
```

然后在配置文件中添加以下内容：

- Windows：`%APPDATA%\Claude\claude_desktop_config.json`
- macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "danbooru-searcher": {
      "command": "mcp-remote",
      "args": [
        "https://sakizuki-danboorusearch.hf.space/mcp/mcp"
      ]
    }
  }
}
```

保存后重启 Claude Desktop，工具列表中出现 `search_tags`、`get_related_tags`、`get_artist_recommendations`、`get_anima_format` 和 `get_newbie_format` 即为成功。

> 注意：不要使用 `"url"` 字段直接填写地址，Claude Desktop 不支持该格式，会提示配置无效。也不推荐通过 `npx mcp-remote` 调用，首次执行时 `npx` 需要临时下载包，npm 缓存损坏时会导致启动失败。全局安装可规避此问题。

**方法二：图形界面（Cherry Studio 等）**

在 MCP 服务器管理页面点击「添加」，填写以下信息：

- 名称：`danbooru-searcher`（可自定义）
- 类型：`Streamable HTTP`
- URL：`https://sakizuki-danboorusearch.hf.space/mcp/mcp`

> 注意：类型务必选择 **Streamable HTTP**，不要选 SSE，否则工具列表无法正常加载。

添加完成后重启客户端，或在 MCP 管理界面手动刷新连接。

#### 可用工具

接入后 AI 可调用以下工具：

**`search_tags`** — 自然语言搜索标签

接受中文或英文描述，返回匹配的 Danbooru 标签列表和可直接用于 AI 绘画的 prompt 字符串。支持参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `query` | — | 自然语言描述（必填，推荐中文）。支持空格/逗号/顿号显式分隔概念 |
| `search_mode` | `"full_scene"` | 检索策略：`full_scene`、`concept_explore`、`subject_describe`、`precise_lookup` |
| `category` | `"all"` | 标签类别筛选：`"all"` 全部、`"general"` 通用、`"character"` 角色、`"copyright"` 作品 |
| `show_nsfw` | `true` | 是否包含 NSFW 标签 |
| `include_wiki` | `false` | 是否在结果中附带 Danbooru wiki 释义 |

`search_mode` 的选择规则：

- `full_scene`：默认值。用户描述的是一张具体画面时使用，即使描述里有很多元素，也仍然属于完整画面。
- `concept_explore`：用户想浏览某个概念下有哪些候选标签时使用，结果更多，token 消耗更高。
- `subject_describe`：只描述一个单一视觉概念、角色或物品时使用；多元素画面不要用这个模式。
- `precise_lookup`：用于中文或英文的单一概念、拼写纠错及接近精确的标签查询；仍使用语义搜索，并用官方 Tag Alias 规范化返回结果。

返回 JSON 包含：`prompt`（逗号分隔标签）、`keywords`（检索关键词）、`results`（标签结果）。每条结果包含 `tag`、`cn_name`；发生官方 Alias 纠正时包含 `alias_from`，当 `include_wiki=true` 时包含 `wiki`。

**`get_related_tags`** — 共现关联推荐

给定一组已选标签，返回在 Danbooru 图库中与之最常共现的相关标签，基于 NPMI 评分，可有效过滤掉仅因自身热度高而频繁出现的标签。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `tags` | — | 标签列表（必填，Danbooru 英文标签名） |
| `limit` | `50` | 最多返回关联标签数 |
| `show_nsfw` | `true` | 是否包含 NSFW 标签 |
| `include_wiki` | `false` | 是否在结果中附带 Danbooru wiki 释义 |

返回字段：`tag`、`cn_name`、`sources`，当 `include_wiki=true` 时包含 `wiki`。如果输入标签拼写错误但可纠正，会返回纠错说明。

**`get_artist_recommendations`** — 推荐擅长画师

给定一组标签，返回擅长绘制这些元素的画师列表，基于标签-画师 NPMI 共现数据排序。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `tags` | — | 标签列表（必填，Danbooru 英文标签名） |
| `limit` | `30` | 最多返回画师数 |
| `min_cooc` | `3` | 单个 (tag, artist) 对的最小共现次数 |
| `show_nsfw` | `true` | 是否包含 NSFW 数据 |

返回字段：`artist`（画师名）、`cooc_count`（共现次数）、`post_count`（作品数）、`sources`（命中标签）、`top_tags`（该画师最常画的 10 个标签，含中文名）

**`get_anima_format`** — 获取 Anima 提示词格式规范

当用户需要把检索到的标签整理为 Anima 模型的 Hybrid 提示词时调用。返回完整格式说明，包括硬标签层、空间叙事层、标签格式化、权重语法、多人物防串扰和中文解释规则。

**`get_newbie_format`** — 获取 NewBie XML 提示词格式规范

当用户需要输出 NewBie 模型 XML 提示词时调用。返回 XML 结构、字段职责、标签处理规则、多人角色分组和 caption 写作要求。

#### 调用示例

接入后，你可以直接用自然语言告诉 AI 你的需求，AI 会自动选择合适的参数调用工具：

> 帮我把「一个穿白色水手服的少女在大雨中奔跑，表情愤怒、流泪，衣服湿透」转换成 Danbooru 标签 prompt

> 我想找「机械义肢」相关的标签，帮我发散一下

> 帮我查一下「命运石之门的助手」对应的角色标签是什么

> 我已经选了 `white_serafuku` 和 `rain` 这两个标签，帮我推荐一些常见的搭配标签

> 帮我找擅长画「平涂风格 + 蓝发少女」的画师

> 查一下「运动社团 校队 比赛 运动会 体育祭 田径部 网球部 篮球部」这些概念对应的标签

> 找「反乌托邦、赛博朋克、蒸汽朋克」的 General 类标签，只查通用特征不查作品名

> 按 Anima 模型格式，把这些标签整理成 Hybrid 提示词

> 按 NewBie XML 格式输出这个角色场景的提示词

#### 服务使用说明

HF Space 在无流量时会进入休眠状态，首次请求需要等待冷启动（约 30~60 秒）。若 AI 调用超时，稍等片刻后重试即可，或先访问 [Space 页面](https://huggingface.co/spaces/SAkizuki/DanbooruSearch) 将其唤醒。

任何用户都可以免费将本服务的API或MCP接口接入自己的私有或公开服务中。

**如果接入公开服务，务必在主页提供链接到本服务主地址的友情链接**：

```
URL: https://huggingface.co/spaces/SAkizuki/DanbooruSearch
简介: 基于语义匹配的Danbooru标签搜索引擎，支持多维匹配与共现关联推荐。
```

---

## 数据库说明

- 数据来源：Danbooru API 抓取
- 中文翻译与语义扩充：由 LLM 辅助完成
- 角色名、作品名：通过二次元垂类数据库 [Bangumi](https://bgm.tv) API 精确查询，尽量避免幻觉
- 仅收录 Danbooru 频数 **≥ 100** 的标签，保证标签实际可用
- 仅覆盖 General（通用特征）、Character（角色）、Copyright（作品）三类标签

**数据库生成代码：** https://github.com/SuzumiyaAkizuki/danbooru-tag-pipeline

---

## 使用注意事项

- 本工具为 AI 辅助工具，检索结果未必 100% 准确
- 查找结果可能包含 NSFW 内容（默认隐藏）
- 仅支持中文 / 英文双语查找
- 如果检索结果不理想，欢迎点击「没搜到？」按钮提交反馈，帮助持续优化引擎

## 支持项目

DanbooruSearch 会继续免费开放。如果这个小工具刚好帮你省了点时间，可以自愿赞赏一点维护成本；赞赏不解锁任何功能、不提高额度、不影响搜索结果，也不会获得特殊待遇。量力而行就好，未成年人请勿赞赏。

![微信赞赏码](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202501120027592.png)


## 搜索统计

![](https://dsocounter.oss-cn-hongkong.aliyuncs.com/danbooru_counter/search_stats.png)

## 友情链接

- 本项目中借鉴了[Good Anima](https://github.com/ShiroEirin/comfyui-good-anima)项目中关于Anima模型提示词书写的相关内容
- [Linux Do社区](https://linux.do/)

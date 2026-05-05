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

## 本地部署

本项目支持在本地运行，无需依赖云端服务。首次启动会自动下载模型和数据文件，后续启动直接使用本地缓存。

### 环境要求

- Python 3.10+
- NVIDIA GPU（推荐，CPU 也可运行但全量编码会很慢）
- 约 5GB 磁盘空间（模型 4.3GB + 数据 410MB）

### 快速开始

1. 克隆仓库并进入目录：

```bash
git clone https://github.com/SuzumiyaAkizuki/DanbooruSearchOnline.git
cd DanbooruSearchOnline
git checkout local-release
```

2. 双击 `start.bat`（Windows），脚本会自动：
   - 检测 GPU 和 CUDA 版本
   - 安装对应版本的 PyTorch（GPU 版或 CPU 版）
   - 安装其余依赖
   - 启动 Web UI

3. 浏览器会自动打开 `http://127.0.0.1:11111`

### 手动安装

如果不用 `start.bat`，可以手动操作：

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 安装 PyTorch（根据你的 CUDA 版本选择，参考 https://pytorch.org/get-started/locally/）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install -r requirements.txt

# 启动
python ui_nicegui.py
```

### 配置文件

项目根目录的 `config.json` 可自定义以下设置：

```json
{
  "model_dir": "my_model_bge_m3",
  "port": 11111,
  "csv_file": "origin_database/tags_enhanced.csv",
  "cooc_file": "origin_database/cooccurrence_clean.csv",
  "atomic_cjk_max_len": 7
}
```

| 字段 | 默认值 | 说明 |
|---|---|---|
| `model_dir` | `my_model_bge_m3` | 本地模型目录名，不存在时自动从 HuggingFace Hub 下载 |
| `port` | `11111` | Web UI 监听端口 |
| `csv_file` | `origin_database/tags_enhanced.csv` | 标签数据库路径 |
| `cooc_file` | `origin_database/cooccurrence_clean.csv` | 共现矩阵路径（引擎会自动查找同名 `.parquet` 文件） |
| `atomic_cjk_max_len` | `7` | 中文分词原子长度阈值，超过此长度的片段会被自动分词 |

### 数据自动更新

引擎每次启动时会检查 `origin_database/` 下的数据文件是否有更新（通过 GitHub ETag 机制），有新版本时自动下载，无变化时跳过。网络不可用时使用本地已有文件，不影响启动。

### GPU 加速注意事项

`pip install torch` 默认安装的是 **CPU 版本**，全量编码时会非常慢。请务必安装对应 CUDA 版本的 PyTorch：

| GPU 系列 | CUDA 版本 | PyTorch 安装命令 |
|---|---|---|
| RTX 5090/5080 等（Blackwell） | 12.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| RTX 4090/4080 等（Ada） | 12.6 | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| RTX 3090/3080 等（Ampere） | 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| RTX 2080/2070 等（Turing） | 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

`start.bat` 会自动检测并安装正确版本。安装完成后可通过以下命令验证 GPU 是否可用：

```python
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 即为成功。

### 本地 MCP 接入（stdio 模式）

本地版使用 stdio 模式运行 MCP 服务，无需启动 Web UI 即可接入 Claude Desktop 等客户端。

在 Claude Desktop 配置文件中添加（Windows：`%APPDATA%\Claude\claude_desktop_config.json`，macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "danbooru-searcher": {
      "command": "python",
      "args": ["mcp_server.py 的完整绝对路径"]
    }
  }
}
```

例如 Windows 上：

```json
{
  "mcpServers": {
    "danbooru-searcher": {
      "command": "python",
      "args": ["D:\\DanbooruSearchOnline\\mcp_server.py"]
    }
  }
}
```

保存后重启 Claude Desktop，工具列表中出现 `search_tags` 和 `get_related_tags` 即为成功。首次调用时引擎会自动初始化（约 1~3 分钟），后续调用秒级响应。

---

## 核心特性

本版本在基础语义检索之上，引入了完整的标签工作流辅助功能：

* **多维度语义匹配：** 底层支持 4 个维度的向量空间检索（英文、中文扩展词、维基释义、中文核心词），大幅提升模糊描述的命中率。
* **智能分词与显式分隔：** 支持一键开启/关闭自动分词。开启后自动提取长句中概念分别检索；同时支持用空格、换行、中文逗号（，）和顿号（、）手动分隔概念，系统会尊重你的分隔意图，将每个片段作为整体匹配（超过 7 字的片段仍然自动分词）。
* **标签互相关推荐：** 引擎内置标签共现数据。当你选中某些标签后，系统会自动推荐在 Danbooru 图库中经常与它们一同出现的相关标签，辅助补全画面细节。
* **精细化查阅与过滤：** 支持按标签类别（General通用 / Character角色 / Copyright作品）进行硬过滤。
  * 提供 **NSFW 保护模式**（开启后自动模糊敏感词汇）。
  * 表格内置 Danbooru 维基释义悬浮提示，并提供前往原站的快捷链接。


## 功能概览

本工具提供以下五项核心功能：

- [精确查词](#1-精确查词)（容错拼写错误）
- [概念模糊发散](#2-概念模糊发散)
- [输入描述查词](#3-输入描述查词)
- [完整画面查找](#4-完整画面查找)
- [关联推荐](#5-关联推荐共现匹配)

前四项基于**语义向量匹配**，后一项基于**标签共现数据**。

---

## 参数说明

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

## 1. 精确查词

**适用场景：** 你知道某个标签的大概写法，但不确定拼写是否准确，或者只记得部分写法。

本工具通过语义相似度匹配而非字符串精确匹配，因此具备一定的拼写容错能力。

**建议参数：** Top K: 20 | 结果上限: 10 | 热度权重: 0.15 | **关闭智能分词**

> 关闭智能分词，确保整个输入作为一个整体进行语义匹配，不会被拆散。

### 示例

| 你的输入 | 实际标准标签 | 说明 |
|---|---|---|
| `selafuku` | `serafuku` | 经典拼写错误，仍可命中 |
| `school unifor` | `school_uniform` | 漏字母，仍可命中 |
| `twintail` | `twintails` | 单复数混淆，仍可命中 |
| `zettai ryouiki` | `zettai_ryouiki` | 罗马音输入，可命中 |
| `thighhigh` | `thighhighs` | 词形变体，仍可命中 |

---

## 2. 概念模糊发散

**适用场景：** 你脑中有一个模糊的概念或风格意象，但不知道在 Danbooru 里对应哪些标签，希望系统帮你"发散"出相关的候选标签。

**建议参数：** Top K: 80~160 | 结果上限: 80 | 热度权重: 0.15 | **开启智能分词**

### 示例

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

## 3. 输入描述查词

**适用场景：** 你能描述某个具体的事物、角色、场景，但不知道 Danbooru 的对应标签。可以使用中文或英文自然语言描述，也可以通过 IP 名称、角色外号来查找角色标签。

**建议参数：** Top K: 20 | 结果上限: 20 | 热度权重: 0.15 | **关闭智能分词**

> 关闭智能分词，使系统对你的整段描述做整体语义匹配，而不是拆词。

### 示例

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

## 4. 完整画面查找

**适用场景：** 你脑中已经有了一幅完整的画面，希望一次性把整段描述转换成尽可能多的 Danbooru 标签，用于 AI 绘画的完整 Prompt。

**建议参数：** Top K: 5 | 结果上限: 80 | 热度权重: 0.15 | **开启智能分词**

> 开启智能分词：系统会自动拆分你的长句，提取其中的关键概念分别检索，再合并去重，覆盖尽可能多的画面元素。

### 示例

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

## 5. 关联推荐（共现匹配）

关联推荐与前四项功能机制不同，它不依赖语义相似度，而是基于 **Danbooru 图库中标签的共现统计**：如果两个标签经常出现在同一张图上，就认为它们相关。

**使用方式：** 在左侧结果表格中勾选你感兴趣的标签，右侧关联推荐栏会自动刷新，显示在 Danbooru 图库中与你选中标签最常共现的其他标签。也可以点击「根据已选刷新」手动触发。

关联推荐的得分是基于 NPMI（互信息归一化）计算的，能有效过滤掉那些只是因为自身热度高而频繁出现的标签。

### 适用场景

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

## 使用技巧

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

## API 接口

本工具同时提供 REST API，可集成到你自己的工作流中。

启动后访问 `/api/docs` 查看完整的交互式 API 文档。

**搜索接口**
```
POST /api/search
{
    "query": "白色水手服的女孩",
    "top_k": 20,
    "limit": 20,
    "show_nsfw": false
}
```

**关联推荐接口**
```
POST /api/related
{
    "tags": ["white_serafuku", "sailor_collar"],
    "limit": 20,
    "show_nsfw": false
}
```

详见：[API 文档](https://sakizuki-danboorusearch.hf.space/api/docs)

---

## MCP 接口

本工具提供 MCP（Model Context Protocol）服务，支持将搜索引擎直接接入 Claude Desktop、Cursor、Cherry Studio 等支持 MCP 协议的大模型客户端，让 AI 能够在对话中直接调用标签搜索能力。

**MCP 服务地址：**
```
https://sakizuki-danboorusearch.hf.space/mcp/mcp
```

### 接入方法

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

保存后重启 Claude Desktop，工具列表中出现 `search_tags` 和 `get_related_tags` 即为成功。

> 注意：不要使用 `"url"` 字段直接填写地址，Claude Desktop 不支持该格式，会提示配置无效。也不推荐通过 `npx mcp-remote` 调用，首次执行时 `npx` 需要临时下载包，npm 缓存损坏时会导致启动失败。全局安装可规避此问题。

**方法二：图形界面（Cherry Studio 等）**

在 MCP 服务器管理页面点击「添加」，填写以下信息：

- 名称：`danbooru-searcher`（可自定义）
- 类型：`Streamable HTTP`
- URL：`https://sakizuki-danboorusearch.hf.space/mcp/mcp`

> 注意：类型务必选择 **Streamable HTTP**，不要选 SSE，否则工具列表无法正常加载。

添加完成后重启客户端，或在 MCP 管理界面手动刷新连接。

### 可用工具

接入后 AI 可调用以下两个工具：

**`search_tags`** — 自然语言搜索标签

接受中文或英文描述，返回匹配的 Danbooru 标签列表和可直接用于 AI 绘画的 prompt 字符串。支持参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `query` | — | 自然语言描述（必填，推荐中文）。支持空格/逗号/顿号显式分隔概念 |
| `use_segmentation` | `true` | 智能分词开关，完整画面描述时开启，精确查词时关闭 |
| `top_k` | `5` | 每个分词的语义召回数量，发散探索时调高至 80~160 |
| `limit` | `80` | 返回标签总数上限 |
| `popularity_weight` | `0.15` | 热度权重，控制高频标签的排名影响 |
| `show_nsfw` | `true` | 是否包含 NSFW 标签 |
| `category` | `"all"` | 标签类别筛选：`"all"` 全部、`"general"` 通用、`"character"` 角色、`"copyright"` 作品 |

**`get_related_tags`** — 共现关联推荐

给定一组已选标签，返回在 Danbooru 图库中与之最常共现的相关标签，基于 NPMI 评分，可有效过滤掉仅因自身热度高而频繁出现的标签。

### 调用示例

接入后，你可以直接用自然语言告诉 AI 你的需求，AI 会自动选择合适的参数调用工具：

> 帮我把「一个穿白色水手服的少女在大雨中奔跑，表情愤怒、流泪，衣服湿透」转换成 Danbooru 标签 prompt

> 我想找「机械义肢」相关的标签，帮我发散一下

> 帮我查一下「命运石之门的助手」对应的角色标签是什么

> 我已经选了 `white_serafuku` 和 `rain` 这两个标签，帮我推荐一些常见的搭配标签

> 查一下「运动社团 校队 比赛 运动会 体育祭 田径部 网球部 篮球部」这些概念对应的标签

> 找「反乌托邦、赛博朋克、蒸汽朋克」的 General 类标签，只查通用特征不查作品名

### 查询写作技巧（写给 LLM）

为了让搜索更精准，在构造 `query` 时请注意以下几点：

1. **概念列表用显式分隔符：** 需要批量查询多个独立概念时，用空格、中文逗号（，）或顿号（、）分隔。每个 ≤7 字的片段会作为完整概念匹配，不会被错误拆开。例如 `运动社团 校队 比赛` 而非 `运动社团校队比赛`。

2. **画面描述写自然语句：** 完整场景描述不需要显式分隔，系统会自动分词。例如 `一个穿着白色水手服蓝色短裙的少女在雨中奔跑`。

3. **查角色/作品时关闭分词：** 精确查找某个角色名或作品名时，设 `use_segmentation=false` 和 `category="character"`（或 `"copyright"`），避免名字中的词汇被单独匹配到无关标签。

4. **长短语会自动保持完整语义：** 即使用显式分隔符切出的片段超过 7 字，系统也会对其自动分词并同时保留原始片段作为额外查询（从句级语义），不会丢失整体语境。

### 注意事项

HF Space 在无流量时会进入休眠状态，首次请求需要等待冷启动（约 30~60 秒）。若 AI 调用超时，稍等片刻后重试即可，或先访问 [Space 页面](https://huggingface.co/spaces/SAkizuki/DanbooruSearch) 将其唤醒。

---

## 数据库说明

- 数据来源：Danbooru API 抓取
- 中文翻译与语义扩充：由 LLM 辅助完成
- 角色名、作品名：通过二次元垂类数据库 [Bangumi](https://bgm.tv) API 精确查询，尽量避免幻觉
- 仅收录 Danbooru 频数 **≥ 100** 的标签，保证标签实际可用
- 仅覆盖 General（通用特征）、Character（角色）、Copyright（作品）三类标签

**数据库生成代码：** https://github.com/SuzumiyaAkizuki/danbooru-tag-pipeline

---

## 注意事项

- 本工具为 AI 辅助工具，检索结果未必 100% 准确
- 查找结果可能包含 NSFW 内容（默认隐藏）
- 仅支持中文 / 英文双语查找
- 如果检索结果不理想，欢迎点击「没搜到？」按钮提交反馈，帮助持续优化引擎



## 搜索统计

![](https://dsocounter.oss-cn-hongkong.aliyuncs.com/danbooru_counter/search_stats.png)

## 友情链接

- [Linux Do社区](https://linux.do/)
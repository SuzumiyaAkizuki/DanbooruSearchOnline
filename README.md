---
title: DanbooruSearch
emoji: 👁
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: 基于语义匹配的 Danbooru 标签搜索引擎，支持多维匹配、智能分词与共现关联推荐。
tags:
  - text-to-image
  - prompt-engineering
  - stable-diffusion
  - danbooru
  - nlp
---

# Danbooru Tags Searcher

本项目提供一种基于语义匹配的 Danbooru 标签查找方案。用户可以通过输入日常语言或模糊描述，查找并匹配 Danbooru 数据集中对应的标准标签，适用于辅助构建 Stable Diffusion、NovelAI 等 AI 绘画工具的提示词（Prompt）。

目前支持使用汉语和英语进行查找。

**立即使用：**https://huggingface.co/spaces/SAkizuki/DanbooruSearch

**ComfyUI 插件版本：** [ComfyUI-DanbooruSearcher](https://github.com/SuzumiyaAkizuki/ComfyUI-DanbooruSearcher)

---

## 核心特性

本版本在基础语义检索之上，引入了完整的标签工作流辅助功能：

* **多维度语义匹配：** 底层支持 4 个维度的向量空间检索（英文、中文扩展词、维基释义、中文核心词），大幅提升模糊描述的命中率。
* **智能分词控制：** 支持一键开启/关闭智能分词。开启时自动提取长句中的概念进行组合检索；关闭后严格执行整句语义匹配。
* **标签互相关推荐：** 引擎内置标签共现数据。当你选中某些标签后，系统会自动推荐在 Danbooru 图库中经常与它们一同出现的相关标签，辅助补全画面细节。
* **精细化查阅与过滤：** 支持按标签类别（General通用 / Character角色 / Copyright作品）进行硬过滤。
  * 提供 **NSFW 保护模式**（开启后自动模糊敏感词汇）。
  * 表格内置 Danbooru 维基释义悬浮提示，并提供前往原站的快捷链接。


## 检索参数说明

你可以通过基础与高级设置精确控制搜索行为：

* **Top K（语义相关）：** 针对单个分词，检索最相关的候选标签数量。
* **结果上限：** 最终展示的标签总数限制（建议设为 70-80 以适配主流 SDXL 模型）。
* **热度权重（0~1）：** 控制标签在 Danbooru 站内发帖量对最终排序的影响。数值越高，越倾向于推荐常用、高频的标签（推荐默认值为 0.15）。


## 工作模式与示例

### 1. 完整画面查找
输入对画面的完整自然语言描述，程序将进行智能分词并生成匹配的标签集。

**示例输入：**
> 一个穿着白色水手服、蓝色短裙的少女，在下着大雨的城市街道奔跑，她的表情是不甘、愤怒、流泪，她的衣服湿透。

**建议参数：** Top K: 5 | 结果上限: 80 | 热度权重: 0.15

![image-20260305233414260](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202603052334299.png)

### 2. 关键词精确查找
适用于对某个概念有模糊印象，但不知晓其准确 Danbooru 英文标签的场景。

**示例输入：**
> 紧身衣勒入皮肤产生的凹陷

**建议参数：** Top K: 5 | 结果上限: 10 | 热度权重: 0.15 | 关闭智能分词

![image-20260305233050518](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202603052331706.png)

### 3. 概念模糊发散
围绕单一基础概念进行搜索，利用系统的“关联推荐”功能发现更多相关的细分搭配标签。

**示例输入：**
> 中国风古装

**建议参数：** Top K: 40 | 结果上限: 80 | 热度权重: 0.15

![image-20260305233520318](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202603052335567.png)

### 4. 活用关联推荐

关联推荐显示两个标签有多大可能同时出现。

你可以利用这个功能来查看具有某种特征的人物

**示例输入：**

> 露指手套

![image-20260305233709615](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202603052337245.png)

可以查看某个人物具有的特征：

**示例输入：**

> 能天使

![image-20260305233802080](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202603052338649.png)

可以查看经常出现在一起的相关标签：

**示例输入：**

> 战斗机

![image-20260305233920236](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202603052339743.png)


---

**⚠️ 注意事项**
* 本网站为 AI 辅助工具，检索结果未必 100% 准确。
* 仅显示 Danbooru 频数超过 100 的标签以保证标签可用性。
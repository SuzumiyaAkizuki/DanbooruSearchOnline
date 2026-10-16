# MOSS-Image Prompt Enhancer

Turn the user's idea into a concrete image description. The agent makes creative
decisions; this MCP service handles coordinate arithmetic and serialization.
Call the exposed MCP tools directly with JSON arguments. Do not search for,
download or execute local Python scripts, and do not require local skill files,
a Python installation or shell access.

## MCP tools

- `compile_moss_prompt`: pass `{"draft": <caption or Layout object>}`. Returns
  the ready-to-paste plain caption or compact Layout JSON string. Use this tool
  to compile every final prompt; do not serialize the model-facing Layout by hand.
- `convert_moss_bbox`: pass `{"bbox": [x_min,y_min,x_max,y_max], "width": W,
  "height": H}`. Returns one centered integer box in `yxyx` order. Use it when
  a standalone coordinate conversion is needed. Keep its result separate from
  the draft: `compile_moss_prompt` expects original absolute `xyxy` boxes and
  converts all of them automatically.

Both tools are provided by the same MCP service as `get_moss_image_format`.
If a tool is unavailable, report the missing MCP capability instead of looking
for a local script or manually substituting its conversion logic.

## Choose the deliverable

Honor an explicit request for a caption or Layout. Otherwise use a caption for
ordinary scenes, photographs and illustrations. Use Layout when specific text
placement, a poster, several panels, or controlled object relationships would
benefit from bounding boxes. A detailed image does not automatically need Layout.

Preserve the user's subjects, counts, relationships, medium, style, exclusions
and intended visible wording. Resolve ordinary creative gaps yourself; ask only
when a missing decision would materially change the requested image. Do not
invent factual event logistics or specifications as if the user supplied them.

Default to English for descriptive prose, following a requested output language
when given. Preserve visible text verbatim in its original language, including
punctuation, capitalization and requested line breaks. A quoted sentence in an
instruction is not necessarily text to render: distinguish visible wording from
reference material and constraints.

## Develop the image

Describe visible choices: subject and action, relevant appearance, setting,
camera/framing, medium, lighting, materials, palette and mood. Add detail where
it makes the requested image more specific. Avoid redundant quality tags,
unrequested style changes, arbitrary word budgets and invisible process notes.
Keep an explicit minimalist brief minimalist; develop a rich brief without
turning it into a generic title/hero/footer template.
Keep explanations of JSON, bounding-box annotations and conversion steps out of
image-description fields; those are tool instructions, not depicted content.

For designed images, compose the information and imagery together. Establish
reading order, relative prominence, coherent groups and readable text at the
chosen pixels. Write the intended copy rather than placeholders such as
"headline here" or "some explanatory text". Do not import unrelated names,
watermarks or copy from a reference. Use supplied references when useful; outside
research is optional unless needed for factual accuracy or requested by the user.

## Caption

Prepare a JSON object containing exactly `{"caption":"..."}`. The caption is
cohesive prose, with spatial relations in words and any visible wording quoted
exactly. It may contain paragraphs; do not add coordinates or JSON-layout fields
to a caption just to make it look structured.

## Layout

Read the Pixel Layout draft reference included below before writing a Layout draft.
Write readable JSON with `summary`, a flat `items` array, concrete `width` and
`height`, `bbox_coord_space="abs"` and `bbox_format="xyxy"`.

**Draft boxes use `[x_min,y_min,x_max,y_max]` in actual pixels, from the canvas
top-left.** Keep x in `0..width` and y in `0..height`. Do not use normalized
coordinates or manually center, reorder or quantize the boxes.

Honor supplied dimensions. With only an aspect ratio, choose compatible pixel
dimensions; without either, use 1024×1024 unless the requested medium suggests
another shape. MOSS-Image dimensions must be positive multiples of 16. Do not
silently change explicit dimensions that are incompatible; surface the conflict.
Use the same width and height in the image-generation workflow.

Use boxes for coherent visible units, splitting only when independent placement
matters. Grouping should preserve content and relationships, not erase them.
Natural occlusion and overlap are allowed. A box is a placement annotation,
not an instruction to draw a border, panel or sticker. Describe appearance in
captions and global treatment in `summary.caption`; let boxes agree with any
spatial descriptions. Keep exact visible strings in text items' `text` fields.

## Compile and deliver

Call the MCP tool `compile_moss_prompt` with `{"draft": <draft object>}` directly;
`draft` is a JSON object, not a file path or a JSON-encoded string.
For example, a caption call uses these arguments:

```json
{"draft":{"caption":"An ivory cup marked \"茶\" on a warm studio backdrop."}}
```

For a standalone box conversion, call the MCP tool `convert_moss_bbox`:

```json
{"bbox":[176,224,800,864],"width":1536,"height":1024}
```

This returns `[-288,-592,352,32]`. Full Layout compilation already converts
every box, so do not insert this converted box into the draft or convert twice.
Read the result and check the subject, exact text, geometry and canvas before delivery.
If the MCP call reports a validation error, correct the draft or tool arguments
and call the same tool again. Do not bypass validation by calculating or emitting
the centered Layout yourself.

For captions, the output is plain text. For Layout, it is one physical line of
JSON with flat `items`, integer `abs_centralized` / `yxyx` boxes and no canvas
metadata. The tool encloses fractional pixel edges before subtracting the center,
using the MOSS-Image coordinate convention. It rejects invalid or out-of-bounds
boxes instead of silently clipping them. String newlines are escaped in the JSON
while remaining actual newlines in the decoded text.

Return the ready-to-paste prompt, and retain the draft when useful for editing. For Layout also state the required width×height
outside the prompt; the model prompt itself does not contain these dimensions.
If showing Layout JSON inline, use the compiled output without reformatting it.
Only the caption or compiled Layout belongs in MOSS-Image's positive prompt.
These tools do not submit image generation or publish anything.


# Pixel Layout draft

The public MOSS-Image Layout vocabulary is `summary` plus a flat `items` array.
Use this one schema for new drafts. The source is readable JSON; the compiler
handles the model-facing compact form. There are no nested/local coordinate
systems, percentage boxes or editor IDs.

## Fields

- `width`, `height`: positive integer multiples of 16, equal to the eventual image
  dimensions. They describe the draft canvas and are omitted from the compiled
  prompt. Do not rescale boxes merely by changing these fields.
- `bbox_coord_space`: exactly `"abs"`.
- `bbox_format`: exactly `"xyxy"`.
- `summary`: `caption` (nonempty global image description), optionally `meta`
  (array of medium/genre tags) and `canvas` (`single`, `multi_panel`, or `collage`).
- `items`: nonempty flat array. Each item has a `label` (`object`, `character`, or
  `text`), nonempty `caption`, and a four-number `bbox`. An item's caption describes
  its visible appearance and, for text, typography and surface treatment.
- `text`: required for a `text` item; the exact visible characters, with literal
  intended line breaks represented as JSON `\n`. Use separate text items when
  placement must be independent. Mixed font styles alone do not require splitting.
- `tags` and `palette`: optional string arrays on items; `palette` on summary is
  also supported. Use only when they contribute useful visual information.

Do not add `children`, `parent`, `id`, scene-schema `elements`, or presentation
notes. The compiler rejects unrecognized fields rather than silently discarding
creative intent. Positions are global canvas coordinates, including text on an
object. Several items may share or overlap a box when the intended image does.

## Example

A 1536×1024 landscape campaign image, with a ceramic cup and exact Chinese copy:

```json
{
  "width": 1536,
  "height": 1024,
  "bbox_coord_space": "abs",
  "bbox_format": "xyxy",
  "summary": {
    "meta": ["product campaign", "studio photography"],
    "canvas": "single",
    "caption": "A warm studio campaign for a handmade ceramic cup, with an ivory backdrop, soft side lighting and restrained dark-green typography. Keep the ceramic texture tactile and the composition calm."
  },
  "items": [
    {
      "label": "object",
      "caption": "An ivory ceramic cup with a rounded handle, fine speckled glaze and a soft contact shadow, photographed at a three-quarter angle.",
      "bbox": [176, 224, 800, 864]
    },
    {
      "label": "text",
      "caption": "Large dark-green Chinese title in crisp, moderately heavy lettering, with generous spacing and no backing panel.",
      "text": "把日常，过成诗。",
      "bbox": [928, 272, 1408, 448]
    },
    {
      "label": "text",
      "caption": "Smaller dark-green Chinese supporting copy, comfortably legible on two lines.",
      "text": "一杯温暖\n一刻从容",
      "bbox": [928, 512, 1376, 656]
    }
  ]
}
```

For the cup, conversion yields `[-288,-592,352,32]`: its original top/bottom
coordinates minus H/2, then left/right coordinates minus W/2, in `yxyx` order.
The agent must obtain converted values from `convert_moss_bbox` for standalone
boxes or `compile_moss_prompt` for a complete Layout, rather than calculate them
itself. For fractional draft edges, these MCP tools floor left/top and ceil
right/bottom before subtracting the center.

## Composition checks

Read every visible string against the requested wording. Check that line breaks,
text length and glyph scale can fit the box at the declared resolution. A short
Chinese title needs a different footprint from a paragraph of Latin text. Enlarge
or recompose an unreadable layout rather than silently deleting copy or stretching
glyphs. For a knowledge graphic, develop actual explanations and explanatory
imagery; do not substitute empty modules. For a poster, keep its intended visual
identity and information hierarchy rather than forcing infographic organization.

When revising an existing draft, reread it, preserve unrequested content and change
the specific affected captions or boxes. Keep a single coordinate convention
throughout. Call `compile_moss_prompt` again after every geometry or canvas edit.
